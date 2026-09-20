# FashionIQ 멀티모달 검색 실험 결과

## 1. 실험 목적

FashionIQ 데이터셋에서 동일한 멀티모달 임베딩 모델을 사용해 다음 검색 전략을 비교하였다.

* **Text-only**: 수정 텍스트만 query embedding으로 사용
* **Image-only**: reference 상품 이미지만 사용
* **Score Fusion**: Text/Image similarity score를 weighted sum
* **Native Multimodal**: reference image와 modification text를 하나의 멀티모달 입력으로 함께 encoding

Score Fusion은 다음과 같이 계산하였다.

$$
S(q,d)
=
\alpha S_{\text{image}}(q,d)
+
(1-\alpha)S_{\text{text}}(q,d)
$$

여기서 \(\alpha\)는 image score의 weight이며, 0.0부터 1.0까지 0.1 단위로 탐색하였다.

---

## 2. 실험 환경

| 항목                  | 설정                            |
| ------------------- | ----------------------------- |
| GPU                 | NVIDIA GeForce RTX 4090       |
| PyTorch             | 2.8.0+cu126                   |
| CUDA                | 12.6                          |
| Model               | Qwen/Qwen3-VL-Embedding-2B    |
| Embedding Dimension | 2048                          |
| Dataset             | FashionIQ Validation          |
| Categories          | Dress, Shirt, Toptee          |
| Total Queries       | 6,016                         |
| Dress               | 2,017 queries / 3,817 gallery |
| Shirt               | 2,038 queries / 6,346 gallery |
| Toptee              | 1,961 queries / 5,373 gallery |

모델 checkpoint 로딩 상태는 정상으로 확인되었다.

```text
missing keys: 0
unexpected keys: 0
mismatched keys: 0
```

Vision encoder의 주요 weight 역시 정상적으로 로드되었다.

```text
model.visual.patch_embed.proj.weight
shape: (1024, 3, 2, 16, 16)
std: 0.008397
finite: True
```

Embedding sanity check 결과 동일 이미지는 거의 1.0의 cosine similarity를 보였고, 서로 다른 이미지는 확연히 낮은 similarity를 보였다.

```text
same-image similarity : 1.000001
different image #1   : 0.548917
different image #2   : 0.282184
```

따라서 이전 실험에서 발생했던 vision weight 초기화 문제는 해결된 것으로 판단된다.

---

# 3. 전체 결과

| Method                | Image Weight α |        R@1 |        R@5 |       R@10 |       R@50 |        MRR |    nDCG@10 |
| --------------------- | -------------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Image-only            |            1.0 |      1.50% |      4.60% |      6.95% |     15.96% |     0.0346 |     0.0379 |
| Text-only             |            0.0 |     12.27% |     23.25% |     29.59% |     49.68% |     0.1825 |     0.1999 |
| Fusion                |            0.1 |     13.43% |     25.83% |     32.63% |     54.31% |     0.2010 |     0.2207 |
| Fusion                |            0.2 |     15.01% |     29.14% |     36.85% |     58.96% |     0.2247 |     0.2487 |
| **Fusion**            |        **0.3** | **16.62%** | **32.51%** | **40.38%** | **61.42%** | **0.2458** | **0.2736** |
| Fusion                |            0.4 |     15.16% |     31.52% |     39.36% |     59.97% |     0.2332 |     0.2617 |
| Fusion                |            0.5 |     10.75% |     24.32% |     31.95% |     52.66% |     0.1790 |     0.2025 |
| Fusion                |            0.6 |      5.90% |     15.79% |     22.36% |     41.21% |     0.1137 |     0.1304 |
| Fusion                |            0.7 |      3.13% |     10.44% |     14.78% |     30.80% |     0.0724 |     0.0823 |
| Fusion                |            0.8 |      2.18% |      7.18% |     10.89% |     23.74% |     0.0519 |     0.0587 |
| Fusion                |            0.9 |      1.65% |      5.54% |      8.39% |     19.18% |     0.0406 |     0.0451 |
| **Native Multimodal** |              — | **18.65%** | **36.10%** | **44.37%** | **64.83%** | **0.2729** | **0.3041** |

---

# 4. 핵심 결과

## 4.1 Native Multimodal이 가장 높은 성능

가장 좋은 결과는 Qwen3-VL에 image와 text를 동시에 입력한 **Native Multimodal** 방식이었다.

```text
R@1     18.65%
R@5     36.10%
R@10    44.37%
R@50    64.83%
MRR      0.2729
nDCG@10  0.3041
```

가장 좋은 weighted score fusion의 R@10 40.38%보다 약 **3.99%p** 높았다.

상대 향상률은 약:

$$
\frac{0.44365-0.40376}{0.40376}
\approx 9.9\%
$$

이다.

MRR에서도

$$
0.2729 \rightarrow 0.2458
$$

로 Native Multimodal이 약 **11% 높은 성능**을 보였다.

즉 image와 text를 독립적으로 검색한 후 score를 조합하는 것보다, **멀티모달 encoder 내부에서 두 modality를 함께 처리하는 것이 추가적인 이득을 제공했다.**

---

## 4.2 Text가 Image보다 훨씬 강한 신호

단일 modality만 사용할 경우:

```text
Text-only  R@10 = 29.59%
Image-only R@10 =  6.95%
```

로 Text-only가 Image-only보다 매우 높은 성능을 보였다.

FashionIQ query text는 reference 이미지에 대한 상대적인 수정 정보를 포함한다.

예를 들어:

```text
"has shorter sleeves"
"is more colorful"
```

따라서 target 상품을 찾을 때 시각적 유사성만 사용하는 것보다 **변경해야 할 semantic attribute를 나타내는 text가 매우 중요한 signal**임을 확인할 수 있다.

---

## 4.3 그렇지만 Image도 명확한 complementary signal 제공

Text-only:

```text
R@10 = 29.59%
```

에서 image score를 10~30% 정도 추가하면:

```text
α=0.1 → 32.63%
α=0.2 → 36.85%
α=0.3 → 40.38%
```

까지 지속적으로 상승했다.

즉 image 정보가 단독으로는 약하지만, text와 함께 사용할 때는 상당한 추가 정보를 제공한다.

특히 최적 weighted fusion은:

$$
0.3S_{image}+0.7S_{text}
$$

였다.

따라서 FashionIQ에서는 대략적으로:

```text
Text signal  > Image signal

하지만

Text + Image > Text
```

의 관계가 명확하다.

---

# 5. Fusion weight 분석

R@10 기준으로 보면 다음과 같은 형태다.

```text
Image weight α

0.0  Text only       29.59%
0.1                  32.63%
0.2                  36.85%
0.3  ★ Best Fusion   40.38%
0.4                  39.36%
0.5                  31.95%
0.6                  22.36%
0.7                  14.78%
0.8                  10.89%
0.9                   8.39%
1.0  Image only       6.95%
```

성능은 α≈0.3에서 peak를 형성한다.

즉 단순 late fusion을 사용한다면 실험상 적절한 비율은:

$$
Image : Text \approx 3 : 7
$$

이었다.

특히 0.3에서 0.4까지는 성능 차이가 크지 않지만, image weight가 0.5 이상으로 증가하면 급격히 성능이 감소한다.

이는 FashionIQ에서는 **text modification을 중심 signal로 두고, reference image는 제품의 기본 형태와 identity를 보완하는 signal로 사용하는 구조**가 적절하다는 것을 보여준다.

---

# 6. Native Multimodal vs Weighted Fusion

가장 중요한 비교는 다음이다.

| Metric  | Best Score Fusion | Native Multimodal | Difference |
| ------- | ----------------: | ----------------: | ---------: |
| R@1     |            16.62% |        **18.65%** |    +2.03%p |
| R@5     |            32.51% |        **36.10%** |    +3.59%p |
| R@10    |            40.38% |        **44.37%** |    +3.99%p |
| R@50    |            61.42% |        **64.83%** |    +3.41%p |
| MRR     |            0.2458 |        **0.2729** |    +0.0271 |
| nDCG@10 |            0.2736 |        **0.3041** |    +0.0306 |

두 방식의 차이는 다음과 같다.

### Score Fusion

```text
Image
  ↓
Image embedding ── similarity ──┐
                                ├─ weighted sum
Text                            │
  ↓                             │
Text embedding ─── similarity ──┘
```

Image와 Text가 서로 독립적으로 encoding된다.

### Native Multimodal

```text
Reference Image
       +
Modification Text
       ↓
Qwen3-VL multimodal encoder
       ↓
Joint embedding
       ↓
Image gallery retrieval
```

두 modality가 encoder 내부에서 interaction한 이후 하나의 representation을 만든다.

실험에서는 두 번째 방식이 consistently 더 높은 성능을 기록하였다.

따라서 단순히 두 modality의 검색 결과를 섞는 것 이상의 **cross-modal interaction 효과가 존재한다**고 해석할 수 있다.

---

# 7. 최종 비교

전체적인 순위는 다음과 같다.

```text
Native Multimodal
      44.37% R@10
         │
         ▼
Image/Text Score Fusion
      40.38%
         │
         ▼
Text-only
      29.59%
         │
         ▼
Image-only
       6.95%
```

Native Multimodal은 Text-only 대비:

$$
44.37 - 29.59
=
+14.78\%p
$$

향상되었다.

상대 향상률로는 약:

$$
\frac{44.37}{29.59}-1
\approx 50\%
$$

이다.

Image-only 대비로는 약 **6.4배 높은 R@10**을 기록했다.

---

# 8. 결론

이번 FashionIQ 6,016-query 실험에서는 **Qwen3-VL-Embedding-2B의 Native Multimodal Encoding이 가장 높은 검색 성능**을 보였다.

핵심 결과를 요약하면 다음과 같다.

1. **Text-only가 Image-only보다 훨씬 강했다.**

   * R@10: 29.59% vs 6.95%

2. **Image는 Text와 함께 사용할 때 중요한 complementary signal이었다.**

   * Text-only R@10: 29.59%
   * Best weighted fusion: 40.38%

3. **Weighted fusion의 최적점은 Image 30% + Text 70%였다.**

4. **Native Multimodal이 weighted fusion보다도 높은 성능을 기록했다.**

   * Weighted Fusion: R@10 40.38%
   * Native Multimodal: R@10 44.37%

5. 따라서 composed product retrieval에서는 단순한 modality별 검색과 score fusion도 효과적이지만, **image와 text를 encoder 내부에서 함께 처리하는 multimodal representation이 추가적인 성능 향상을 제공하였다.**

실서비스 관점에서는 다음 두 구조 모두 유효하다.

```text
[성능 우선]

Image + Text
    ↓
Multimodal Encoder
    ↓
ANN Search
```

```text
[유연성 / 운영성 우선]

Text Retriever ──┐
                 ├─ Dynamic Weighting
Image Retriever ─┘
        ↓
      Ranking
```

이번 실험에서는 **Native Multimodal 구조가 최대 검색 정확도를 제공했으며**, 개별 modality를 별도로 운영해야 하는 경우에는 **Text 70% / Image 30% 수준의 fusion이 강한 baseline**으로 확인되었다.
