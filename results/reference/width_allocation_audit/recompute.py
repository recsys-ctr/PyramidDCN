"""Recompute descriptive allocation comparisons from 33 frozen historical runs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics as st


ROOT=Path(__file__).resolve().parent
DEFAULTS={'lr':.001,'emb_lr_mult':1.,'sched':None,'train_frac':1.,'ln':True,
          'cross':'dcn','res_mode':'same','reread':'input','state_res':False,
          'res':True,'readout':'last','heads':1,'field_ln':False,'se':False}


def balanced(width,dims):
    usage=[0]*width;result=[]
    for dim in dims:
        selected=[]
        while len(selected)<dim:
            pool=[j for j in range(width) if usage[j]==min(usage)]
            need=dim-len(selected)
            take=pool if need>=len(pool) else [pool[j*len(pool)//need] for j in range(need)]
            selected.extend(take)
            for j in take:usage[j]+=1
        result.append(sorted(selected))
    return result


def metric(values):
    return {'mean':st.mean(values),'sample_SD':st.stdev(values)}


def compute():
    provenance=json.loads((ROOT/'provenance.json').read_text())
    seeds=provenance['seed_order']
    records={}
    for line in (ROOT/'runs.jsonl').read_text().splitlines():
        r=json.loads(line);name=r['name'];receipt=provenance['runs'][name]
        assert name not in records
        assert hashlib.sha256(line.encode()).hexdigest()==receipt['record_sha256']
        assert '/' not in receipt['source_basename'] and '\\' not in receipt['source_basename']
        for key,value in receipt['arm_config'].items():
            assert (value is True and isinstance(r['erank'],dict)) if key=='erank' else r[key]==value
        assert {key:r.get(key,value) for key,value in DEFAULTS.items()}==DEFAULTS
        assert r['model']=='pml_mv' and r['rows']==[0,0,0] and r['views']==1
        assert r['max_epochs']==r['best_epoch']==1 and len(r['history'])==1
        assert r['history'][0]['lr']==.001 and r['batch']==4096
        assert r['emb_reg']==0 and r['clip']==10. and r['emb_init_std']==.0001
        records[name]=r
    assert len(records)==len(provenance['runs'])==33
    summary={'scope':'Historical one-epoch three-run descriptive comparisons; no significance or equivalence claim.',
             'delta_convention':'halving reference minus comparator; positive AUC and negative LogLoss favor the reference.',
             'datasets':{}}
    csv_rows=[]
    for ds in provenance['dataset_order']:
        meta=provenance['datasets'][ds]
        groups={}
        for label,key in [('reference','reference_arm'),('constant','constant_arm'),('reverse','reverse_arm')]:
            rs=[records.get(f'fair_{ds}_{meta[key]}_s{seed}') for seed in seeds]
            if any(r is None for r in rs):
                assert label=='constant' and ds=='avazu'
                continue
            first=rs[0];W=first['input_proj'];dims=first['dims']
            assert first['anchor']==('stagger' if label=='reference' else 'balanced')
            if label=='reference':
                assert dims==[W,W//2,W//4]
                stagger=[list(range(W)),list(range(0,W,2)),list(range(1,W,4))]
                assert balanced(W,dims)==stagger
            core=[W*dims[0],dims[0]*dims[1],dims[1]*dims[2],dims[2]]
            projection=meta['field_count']*meta['field_embedding_dimension']*W
            for r in rs:
                assert r['dims']==dims and r['input_proj']==W
                assert r['serving_linear_macs']==projection+sum(core)
                assert r['serving_linear_flops_2x']==2*r['serving_linear_macs']
                assert r['dense_params']==sum(core)+3*sum(dims)+1
            groups[label]={'widths':dims,'anchor':first['anchor'],'projected_input_width':W,
                           'total_matrix_MAC':first['serving_linear_macs'],
                           'total_matrix_FLOPs':first['serving_linear_flops_2x'],
                           'projection_MAC':projection,'state_map_and_head_MAC':core,
                           'first_map_share_of_core_MAC':core[0]/sum(core),
                           'metrics':{key:metric([r[key] for r in rs]) for key in ['valid_auc','valid_logloss','test_auc','test_logloss']},
                           'run_names':[r['name'] for r in rs]}
        reference=[records[name] for name in groups['reference']['run_names']]
        for label,g in groups.items():
            g['FLOPs_ratio_to_reference']=g['total_matrix_FLOPs']/groups['reference']['total_matrix_FLOPs']
            if label=='reference':continue
            controls=[records[name] for name in g['run_names']]
            g['paired_deltas']={key:dict(metric([x[key]-y[key] for x,y in zip(reference,controls)]),
                                        per_seed=[{'seed':seed,'delta':x[key]-y[key]} for seed,x,y in zip(seeds,reference,controls)])
                               for key in ['valid_auc','valid_logloss','test_auc','test_logloss']}
            for x,y in zip(reference,controls):
                a,b=provenance['runs'][x['name']],provenance['runs'][y['name']]
                csv_rows.append({'dataset':ds,'control':label,'seed':x['seed'],
                    'reference_run':x['name'],'control_run':y['name'],
                    'reference_widths':str(x['dims']),'control_widths':str(y['dims']),
                    'reference_test_AUC':x['test_auc'],'control_test_AUC':y['test_auc'],
                    'delta_test_AUC':x['test_auc']-y['test_auc'],
                    'reference_test_LogLoss':x['test_logloss'],'control_test_LogLoss':y['test_logloss'],
                    'delta_test_LogLoss':x['test_logloss']-y['test_logloss'],
                    'reference_valid_AUC':x['valid_auc'],'control_valid_AUC':y['valid_auc'],
                    'reference_valid_LogLoss':x['valid_logloss'],'control_valid_LogLoss':y['valid_logloss'],
                    'reference_total_matrix_MAC':x['serving_linear_macs'],'control_total_matrix_MAC':y['serving_linear_macs'],
                    'reference_source_basename':a['source_basename'],'reference_source_line':a['source_line_one_based'],
                    'reference_record_sha256':a['record_sha256'],'reference_source_file_sha256':a['historical_source_file_sha256'],
                    'control_source_basename':b['source_basename'],'control_source_line':b['source_line_one_based'],
                    'control_record_sha256':b['record_sha256'],'control_source_file_sha256':b['historical_source_file_sha256']})
        summary['datasets'][ds]={'groups':groups,'balanced_constant_completed':meta['constant_completed']}
    csv_buffer=io.StringIO(newline='')
    writer=csv.DictWriter(csv_buffer,fieldnames=list(csv_rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(csv_rows)
    return json.dumps(summary,indent=2)+'\n',csv_buffer.getvalue(),len(records),len(csv_rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Regenerate the deterministic summary files; update checksums separately.')
    args=parser.parse_args()
    summary,csv_text,runs,pairs=compute()
    outputs={'summary.json':summary,'per_seed.csv':csv_text}
    if args.write:
        for name,text in outputs.items():(ROOT/name).write_text(text)
    else:
        for name,text in outputs.items():assert (ROOT/name).read_text()==text,name
        for line in (ROOT/'SHA256SUMS').read_text().splitlines():
            digest,relative=line.split('  ',1)
            path=(ROOT/relative).resolve();assert path.is_relative_to(ROOT)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,relative
    print(json.dumps({'status':'PASS','historical_runs_checked':runs,'paired_comparisons_recomputed':pairs,
                      'written':bool(args.write),'statistical_significance_claims':False}))


if __name__=='__main__':main()
