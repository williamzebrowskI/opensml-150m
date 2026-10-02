"""Opt-in packed SwiGLU Metal kernels with an explicit reverse-mode derivative."""
from functools import lru_cache
import mlx.core as mx


@lru_cache(maxsize=1)
def kernels():
    forward=mx.fast.metal_kernel(name='sml_swiglu_packed_forward',input_names=['z'],output_names=['h'],source=r'''
        uint start=thread_position_in_grid.x*4;
        for(uint lane=0;lane<4;lane++){
        uint i=start+lane;
        if(i<N){
            uint row=i/H, col=i%H;
            T g=z[row*(2*H)+col], u=z[row*(2*H)+H+col];
            T s=T(1.0f/(1.0f+metal::exp(-float(g))));
            T silu=T(g*s);
            h[i]=T(silu*u);
        }
        }
    ''')
    backward=mx.fast.metal_kernel(name='sml_swiglu_packed_backward',input_names=['z','dh'],output_names=['dz'],source=r'''
        uint start=thread_position_in_grid.x*4;
        for(uint lane=0;lane<4;lane++){
        uint i=start+lane;
        if(i<N){
            uint row=i/H, col=i%H;
            T g=z[row*(2*H)+col], u=z[row*(2*H)+H+col], d=dh[i];
            T s=T(1.0f/(1.0f+metal::exp(-float(g))));
            T silu=T(g*s);
            T dsilu=T(d*u);
            T direct=T(dsilu*s);
            T ds=T(dsilu*g);
            T complement=T(T(1)-s);
            T siggrad=T(s*complement);
            dz[row*(2*H)+col]=T(direct+T(ds*siggrad));
            dz[row*(2*H)+H+col]=T(d*silu);
        }
        }
    ''')
    return forward,backward


@mx.custom_function
def packed_swiglu(z):
    if z.ndim<1 or z.shape[-1]<2 or z.shape[-1]%2:
        raise ValueError('Expected a nonempty packed gate/up last dimension')
    if z.dtype not in (mx.float32,mx.float16,mx.bfloat16):
        raise ValueError('Expected floating-point gate/up activations')
    h=z.shape[-1]//2
    forward,_=kernels()
    return forward(inputs=[z],template=[('T',z.dtype),('H',h),('N',z.size//2)],
                   grid=((z.size//2+3)//4,1,1),threadgroup=(256,1,1),
                   output_shapes=[(*z.shape[:-1],h)],output_dtypes=[z.dtype])[0]


@packed_swiglu.vjp
def packed_swiglu_vjp(primals,cotangent,output):
    z=primals
    _,backward=kernels()
    dz,=backward(inputs=[z,cotangent],template=[('T',z.dtype),('H',z.shape[-1]//2),('N',z.size//2)],
                 grid=((z.size//2+3)//4,1,1),threadgroup=(256,1,1),output_shapes=[z.shape],output_dtypes=[z.dtype])
    return dz
