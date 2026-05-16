extern void encoder_forward(
    float *out, int *inp, float *wte, float *wpe, int B, int T, int C);
extern void encoder_backward(
    float *dwte, float *dwpe, float *dout, int *inp, int B, int T, int C);
extern void layernorm_forward(float *out,
    float *mean,
    float *rstd,
    float *inp,
    float *weight,
    float *bias,
    int B,
    int T,
    int C);
extern void layernorm_backward(float *dinp,
    float *dweight,
    float *dbias,
    float *dout,
    float *inp,
    float *weight,
    float *mean,
    float *rstd,
    int B,
    int T,
    int C);
extern void matmul_forward_naive(float *out,
    const float *inp,
    const float *weight,
    const float *bias,
    int B,
    int T,
    int C,
    int OC);
extern void matmul_forward(float *out,
    const float *inp,
    const float *weight,
    const float *bias,
    int B,
    int T,
    int C,
    int OC);
extern void matmul_backward(float *dinp,
    float *dweight,
    float *dbias,
    const float *dout,
    const float *inp,
    const float *weight,
    int B,
    int T,
    int C,
    int OC);
extern void attention_forward(
    float *out, float *preatt, float *att, float *inp, int B, int T, int C, int NH);
extern void attention_backward(float *dinp,
    float *dpreatt,
    float *datt,
    float *dout,
    float *inp,
    float *att,
    int B,
    int T,
    int C,
    int NH);
extern void gelu_forward(float *out, float *inp, int N);
extern void gelu_backward(float *dinp, float *inp, float *dout, int N);
extern void residual_forward(float *out, float *inp1, float *inp2, int N);
extern void residual_backward(float *dinp1, float *dinp2, float *dout, int N);
extern void softmax_forward(float *probs, float *logits, int B, int T, int V, int Vp);
extern void crossentropy_forward(
    float *losses, float *probs, int *targets, int B, int T, int Vp);
extern void crossentropy_softmax_backward(float *dlogits,
    float *dlosses,
    float *probs,
    int *targets,
    int B,
    int T,
    int V,
    int Vp);
