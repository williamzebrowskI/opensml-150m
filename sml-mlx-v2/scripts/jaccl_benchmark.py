"""V2 distributed worker environment setup."""
import os
from pathlib import Path
import configure_rdma_mesh as mesh
ROOT = mesh.ROOT
def launch_environment(control_transport, backend, port, output):
    from sml_v2.cluster.jaccl_ring_transport import MODE, PORT_ENV, JOB_ENV
    env = dict(os.environ, PATH=str(ROOT / 'cluster/jaccl/production_bin') + ':' + os.environ.get('PATH', ''))
    env['SML_JACCL_CONTROL_TRANSPORT'] = control_transport
    for name in ('MLX_RANK', 'MLX_WORLD_SIZE', 'MLX_HOSTFILE', 'MLX_IBV_DEVICES',
                 'MLX_JACCL_COORDINATOR', 'JACCL_COORDINATOR', 'MLX_METAL_FAST_SYNCH', 'MLX_JACCL_RING',
                 'JACCL_RING', 'JACCL_RANK', 'JACCL_IBV_DEVICES', 'PYTHONPATH', PORT_ENV, JOB_ENV):
        env.pop(name, None)
    if control_transport == MODE:
        if backend != 'jaccl-ring' or not port or not Path(output).is_absolute():
            raise ValueError('Wired ring benchmark requires a ring, port and absolute job directory')
        env[PORT_ENV], env[JOB_ENV] = str(port), str(output)
    return env
