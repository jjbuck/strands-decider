"""psum.py: print the key summary numbers of a neuron-profile full json.  python psum.py DIR_OR_JSON [more keys...]"""
import sys, json, glob, os
p = sys.argv[1]
if os.path.isdir(p): p = glob.glob(f'{p}/json_reports/*.json')[0]
d = json.load(open(p))
s = d['summary'][0] if isinstance(d.get('summary'), list) else d['summary']
keys = ['total_time', 'tensor_engine_active_time_percent', 'vector_engine_active_time_percent', 'scalar_engine_active_time_percent',
        'gpsimd_engine_active_time_percent', 'dma_active_time_percent', 'mfu_estimated_percent', 'hardware_flops', 'transpose_flops',
        'tensor_engine_instruction_count', 'vector_engine_instruction_count', 'scalar_engine_instruction_count', 'gpsimd_engine_instruction_count',
        'psum_read_sbuf_write_count', 'sbuf_read_psum_write_count', 'spill_reload_bytes', 'spill_save_bytes', 'dma_transfer_total_bytes', 'event_count']
keys += sys.argv[2:]
out = {k: s.get(k) for k in keys if k in s}
if len(sys.argv) > 2 and sys.argv[2] == 'ALL':
    out = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
print(json.dumps(out, indent=0))
