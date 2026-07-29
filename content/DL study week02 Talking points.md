# Diffusion Model
## Slide 1 — Key Idea

- Diffusion은 VAE의 아이디어를 확장한 생성 모델이다.

- VAE는 Encoder를 통해 하나의 Latent Variable $z$를 학습한다.

- Diffusion은 데이터를 한 번에 압축하지 않고 점진적으로 Noise를 추가한다.

- 이후 Noise를 한 단계씩 제거하며 원본 데이터를 복원한다.

  

---

  

## Slide 2 — VAE vs Diffusion

  

- VAE: Encoder → Latent $z$ → Decoder

- Diffusion: Forward Noising → Reverse Denoising

- Learned Encoder 대신 Fixed Gaussian Noising Process 사용

- Single Decoder 대신 여러 단계의 Denoising 수행

  

---

  

## Slide 3 — Latent Variable

  

- VAE는 하나의 Latent Variable $z$를 학습한다.

- Diffusion은 $x_1\,\ldots\,x_T$ 전체가 Latent Variable 역할을 한다.

- 즉, Latent Space를 시간축(Time Axis)으로 확장한 모델이다.

  

---

  

## Slide 4 — Forward Process

  

- 원본 이미지에 Gaussian Noise를 조금씩 추가한다.

- 각 단계는 Markov Chain을 따른다.

- 충분한 단계가 지나면 순수 Gaussian Noise가 된다.

- Learned Encoder 대신 사용하는 고정된 확률 과정이다.

  

---

  

## Slide 5 — Reverse Process

  

- 순수 Gaussian Noise에서 시작한다.

- Neural Network가 각 단계의 Noise를 예측한다.

- Noise를 반복적으로 제거하여 원본 이미지를 생성한다.

- Learned Decoder를 여러 단계로 확장한 과정이다.

  

---

  

## Slide 6 — VAE와의 수학적 연결



- VAE는 ELBO를 최대화하여 학습한다.

- Diffusion도 동일하게 ELBO를 최적화한다.

- 차이는 하나의 Latent Variable을 Markov Chain으로 확장한 것이다.

- 즉, 수학적 기반은 VAE와 동일하다.

  

---


## Slide 7 — Markov Chain

- VAE:
  - $x \leftrightarrow z$
- Diffusion:

  - $x_0 \leftrightarrow x_1 \leftrightarrow \cdots \leftrightarrow x_T$

  

- 하나의 Latent Variable을 여러 단계의 Latent Variable로 확장한 구조이다.

  

---
## Final Takeaway
- VAE:
  - Learned Encoder + Learned Decoder
- Diffusion:
  - Fixed Gaussian Noising + Learned Denoising
- 둘 다 Latent Variable Generative Model이다.
- 둘 다 Variational Inference와 ELBO를 기반으로 학습한다.

  

  

# Llama: Transformer에서의 주요 개선점 (Engineering)



1. **RoPE (Rotary Position Embedding)**
2. **Layer Normalization → RMSNorm**
3. **Attention 최적화**
   - KV Cache
   - Multi-Query Attention (MQA)

   - Grouped Query Attention (GQA)

4. **Feed Forward 개선**

   - SwiGLU

  

---

# Relative Positional Encoding (RPE)

### 기존 Self-Attention
* attention은 마르코프 그래프 형태로 해석 가능(자세한 내용은 attention.md로 참고 가능하게)
- Attention은 Query와 Key의 의미적 유사도만 고려한다.
- 즉, $q_i^\top k_j$를 기반으로 Attention을 계산한다.
- 하지만 토큰 간 상대적인 거리 정보는 반영하지 못한다.

---

### Relative Positional Encoding

- 상대 위치를 표현하는 임베딩 $a_{ij}=f(i-j)$를 추가한다.
- 의미적 유사도와 상대 위치를 함께 사전확률로 고려한다.
- 따라서

$$
P(\text{attend}_j|q_i)
\rightarrow
P(\text{attend}_j|q_i,a_{ij})
$$

---

### Relative Position이 반영된 Score

- 기존 Score

$$
q_i^\top k_j
$$

- Relative Position 추가

$$
q_i^\top k_j
+
q_i^\top a_{ij}^{K}
$$

- Softmax를 적용하여 Attention Weight를 계산한다.

---

### 결과

- 의미적 유사도(Content Similarity)
- 상대 위치(Relative Position)

두 정보를 함께 이용하여 Attention을 수행한다.

---

# Rotary Position Embedding (RoPE)

### 기존 Relative PE의 한계

- Relative Position Embedding은 위치 정보를 **더하는(Additive)** 방식이다.
- 별도의 Position Embedding을 학습해야 한다.

---

### RoPE의 핵심 아이디어

- Position Embedding을 더하지 않는다.
- Query와 Key를 위치에 따라 회전(Rotation)시킨다.

$$
\tilde q_i = R_iq_i,
\qquad
\tilde k_j = R_jk_j
$$

---

### Rotation 적용

- Query와 Key를 회전한 뒤 Inner Product를 계산한다.

$$
(R_iq_i)^\top(R_jk_j)
$$

- Rotation의 성질을 이용하면

$$
R_i^\top R_j = R_{j-i}
$$

---

### 최종 Attention Score

기존

$$
q_i^\top k_j
+
q_i^\top a_{ij}^{K}
$$

↓

RoPE

$$
q_i^\top R_{j-i}k_j
$$

---

### 핵심 정리

- Relative PE
  - Position 정보를 더한다(Additive).

- RoPE
  - Position 정보를 Query와 Key의 회전에 내재시킨다.

- 결과적으로 상대 위치 정보가 Query-Key Inner Product 자체에 반영된다.

- 별도의 Relative Position Embedding을 학습하지 않아도 된다.

---
# Gradient 안정화 기법 (Normalization)

  

## RMSNorm

  

- LayerNorm보다 계산량이 적다.

- Mean 계산을 제거하고 RMS만 사용한다.

- 학습 안정성을 유지하면서 계산 비용을 줄인다.

  

---

# Attention 최적화

	  

## KV Cache

  

### 아이디어

- 추론 시 매 토큰마다 전체 Attention을 다시 계산하면 비효율적이다.

- 이전 Key와 Value를 KV Cache에 저장하여 재사용한다.

- 새 토큰은 새로운 Query만 계산하고 Cache와 Attention을 수행한다.

- 즉, 전체 Attention Matrix가 아니라 **새로운 Row 하나만 계산**한다.

  

### 효과

- 중복 계산을 제거한다.

- 추론 속도가 크게 향상된다.

  

---

  

## Multi-Query Attention (MQA)

  

### 기존 Multi-Head Attention의 한계

  

- Multi-Head Attention은 Head마다 독립적인 Key와 Value를 가진다.

- 추론 시 모든 Head의 KV Cache를 반복적으로 읽어야 한다.

- 계산보다 **Memory Access**가 병목이 된다.

- GPU는 계산보다 데이터 이동에 더 많은 시간을 사용한다.

  

### MQA의 아이디어

  

- Query만 여러 Head로 유지한다.

- Key와 Value는 모든 Head가 공유한다.

  

### 효과

  

- KV Cache 크기가 크게 감소한다.

- Memory Read가 크게 줄어든다.

- 계산량은 거의 동일하다.

- 결과적으로 추론 속도가 향상된다.

  

---

  

## Grouped Query Attention (GQA)

  

### 문제점

  

- MQA는 메모리는 절약하지만 표현력이 일부 감소한다.

  

### 해결

  

- Head를 여러 Group으로 나눈다.

- 각 Group이 하나의 Key와 Value를 공유한다.

  

### 효과

  

- 메모리 효율과 모델 성능의 균형을 맞춘다.

- 대부분의 최신 LLM(Llama, Gemma, Qwen, Mistral)이 사용하는 구조이다.

  

---

  



  

# Feed Forward 개선

  

## Swish

### 아이디어  

- ReLU는 음수 영역의 정보를 모두 제거한다.

- Swish는 작은 음수 값도 일부 유지한다.

  

### 효과

  

- 정보 손실을 줄인다.

- Gradient 흐름이 더 부드럽다.

- 학습이 더욱 안정적이다.

- 최신 LLM에서는 ReLU 대신 Swish 계열 활성화 함수를 널리 사용한다.

  

---

  

## SwiGLU

  

### 아이디어

  

- GLU는 Gate를 이용해 중요한 정보만 통과시킨다.

- SwiGLU는 Sigmoid Gate 대신 Swish를 사용한다.

- 입력을 두 부분으로 나누어

  - 한쪽은 Gate

  - 한쪽은 Feature로 사용한다.

  

### 효과

  

- Element-wise 곱을 통해 중요한 정보만 전달한다.

- 표현력이 향상된다.

- 최신 LLM의 Feed Forward Network에서 널리 사용된다.

  

---

  

# 핵심 정리

  

### Attention

  

- KV Cache → 이전 K, V 재사용

- MQA → 모든 Head가 하나의 K, V 공유

- GQA → Group 단위로 K, V 공유

  

### Feed Forward

  

- ReLU → Swish

- GLU → SwiGLU

  

### Normalization

  

- LayerNorm → RMSNorm

  

➡️ 대부분의 개선은 **모델 구조를 크게 바꾸기보다 추론 속도, 메모리 효율, 학습 안정성을 높이기 위한 엔지니어링 개선**이다.