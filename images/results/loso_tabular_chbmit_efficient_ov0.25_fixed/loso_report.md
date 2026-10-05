# LOSO Foundation-Model Report — chbmit

Generated: 2026-10-04T02:45:36

## Configuration

| Parameter | Value |
|---|---|
| Dataset | chbmit |
| Data dir | data/raw/csv-data-v2 |
| Features cache | data/processed/tsfresh_features_chbmit_efficient |
| Models requested | TabICL, TabPFNv3, TabFM, Mitra |
| Window | 30s (3000 samples @ 100Hz), overlap 25% |
| tsfresh feature set | EfficientFCParameters (~782 stats/channel) |
| Seizure threshold | 0.5 |
| Max features (SelectKBest k) | 200 |
| Max in-context rows (TabFM only, all positives kept) | 20000 |
| Mitra context | full train set; halved by Mitra on OOM until it fits, all positives kept (see log for the size used per fold) |
| Mitra fine-tuning | disabled (pure in-context learning) |
| Min channel coverage | 0.5 |
| Seed | 42 |

## Per-patient window / seizure counts

| Patient | Windows | Seizure windows (t+1) | Seizure rate |
|---|---|---|---|
| chb01 | 3726 | 19 | 0.5% |
| chb02 | 2930 | 7 | 0.2% |
| chb03 | 3524 | 18 | 0.5% |
| chb04 | 3611 | 18 | 0.5% |
| chb05 | 3392 | 26 | 0.8% |
| chb06 | 2222 | 5 | 0.2% |
| chb07 | 1620 | 16 | 1.0% |
| chb08 | 1922 | 43 | 2.2% |
| chb09 | 1807 | 13 | 0.7% |
| chb10 | 2459 | 22 | 0.9% |
| chb11 | 2937 | 37 | 1.3% |
| chb12 | 2358 | 46 | 2.0% |
| chb13 | 3107 | 24 | 0.8% |
| chb14 | 2506 | 7 | 0.3% |
| chb16 | 1659 | 0 | 0.0% |
| chb17 | 1742 | 14 | 0.8% |
| chb18 | 3103 | 14 | 0.5% |
| chb19 | 2415 | 11 | 0.5% |
| chb20 | 2548 | 13 | 0.5% |
| chb21 | 2863 | 8 | 0.3% |
| chb22 | 2597 | 10 | 0.4% |
| chb23 | 1331 | 20 | 1.5% |
| chb24 | 2340 | 25 | 1.1% |

## Per-fold metrics

| Model | Patient | Accuracy | Precision | Recall | F1 | F1 Macro | ROC AUC |
|---|---|---|---|---|---|---|---|
| TabICL | chb01 | 0.9946 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.6820 |
| TabICL | chb02 | 0.9976 | 0.0000 | 0.0000 | 0.0000 | 0.4994 | 0.7872 |
| TabICL | chb03 | 0.9949 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.8452 |
| TabICL | chb04 | 0.9956 | 0.7500 | 0.1667 | 0.2727 | 0.6353 | 0.8178 |
| TabICL | chb05 | 0.9932 | 0.8000 | 0.1538 | 0.2581 | 0.6273 | 0.8983 |
| TabICL | chb06 | 0.9977 | 0.0000 | 0.0000 | 0.0000 | 0.4994 | 0.6650 |
| TabICL | chb07 | 0.9920 | 1.0000 | 0.1875 | 0.3158 | 0.6559 | 0.8443 |
| TabICL | chb08 | 0.9781 | 1.0000 | 0.0233 | 0.0455 | 0.5172 | 0.6499 |
| TabICL | chb09 | 0.9934 | 1.0000 | 0.0769 | 0.1429 | 0.5698 | 0.8474 |
| TabICL | chb10 | 0.9919 | 1.0000 | 0.0909 | 0.1667 | 0.5813 | 0.8966 |
| TabICL | chb11 | 0.9874 | 0.0000 | 0.0000 | 0.0000 | 0.4968 | 0.3374 |
| TabICL | chb12 | 0.9805 | 0.0000 | 0.0000 | 0.0000 | 0.4951 | 0.7415 |
| TabICL | chb13 | 0.9923 | 0.0000 | 0.0000 | 0.0000 | 0.4981 | 0.6480 |
| TabICL | chb14 | 0.9972 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.6358 |
| TabICL | chb17 | 0.9908 | 0.0000 | 0.0000 | 0.0000 | 0.4977 | 0.8164 |
| TabICL | chb18 | 0.9955 | 0.0000 | 0.0000 | 0.0000 | 0.4989 | 0.8814 |
| TabICL | chb19 | 0.9959 | 0.6667 | 0.1818 | 0.2857 | 0.6418 | 0.8161 |
| TabICL | chb20 | 0.9945 | 0.0000 | 0.0000 | 0.0000 | 0.4986 | 0.7596 |
| TabICL | chb21 | 0.9972 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.8111 |
| TabICL | chb22 | 0.9961 | 0.0000 | 0.0000 | 0.0000 | 0.4990 | 0.8777 |
| TabICL | chb23 | 0.9850 | 0.0000 | 0.0000 | 0.0000 | 0.4962 | 0.9410 |
| TabICL | chb24 | 0.9876 | 0.0000 | 0.0000 | 0.0000 | 0.4969 | 0.5658 |
| TabPFNv3 | chb01 | 0.9946 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.6468 |
| TabPFNv3 | chb02 | 0.9976 | 0.0000 | 0.0000 | 0.0000 | 0.4994 | 0.7263 |
| TabPFNv3 | chb03 | 0.9949 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.8966 |
| TabPFNv3 | chb04 | 0.9953 | 1.0000 | 0.0556 | 0.1053 | 0.5515 | 0.8506 |
| TabPFNv3 | chb05 | 0.9932 | 0.8000 | 0.1538 | 0.2581 | 0.6273 | 0.8739 |
| TabPFNv3 | chb06 | 0.9973 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.5697 |
| TabPFNv3 | chb07 | 0.9901 | 0.0000 | 0.0000 | 0.0000 | 0.4975 | 0.8809 |
| TabPFNv3 | chb08 | 0.9776 | 0.0000 | 0.0000 | 0.0000 | 0.4943 | 0.6086 |
| TabPFNv3 | chb09 | 0.9928 | 0.0000 | 0.0000 | 0.0000 | 0.4982 | 0.8318 |
| TabPFNv3 | chb10 | 0.9915 | 1.0000 | 0.0455 | 0.0870 | 0.5413 | 0.9016 |
| TabPFNv3 | chb11 | 0.9874 | 0.0000 | 0.0000 | 0.0000 | 0.4968 | 0.3812 |
| TabPFNv3 | chb12 | 0.9805 | 0.0000 | 0.0000 | 0.0000 | 0.4951 | 0.7247 |
| TabPFNv3 | chb13 | 0.9923 | 0.0000 | 0.0000 | 0.0000 | 0.4981 | 0.6033 |
| TabPFNv3 | chb14 | 0.9972 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.6137 |
| TabPFNv3 | chb17 | 0.9908 | 0.0000 | 0.0000 | 0.0000 | 0.4977 | 0.7894 |
| TabPFNv3 | chb18 | 0.9955 | 0.0000 | 0.0000 | 0.0000 | 0.4989 | 0.8863 |
| TabPFNv3 | chb19 | 0.9954 | 0.0000 | 0.0000 | 0.0000 | 0.4989 | 0.8099 |
| TabPFNv3 | chb20 | 0.9949 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.7876 |
| TabPFNv3 | chb21 | 0.9972 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.8092 |
| TabPFNv3 | chb22 | 0.9961 | 0.0000 | 0.0000 | 0.0000 | 0.4990 | 0.9520 |
| TabPFNv3 | chb23 | 0.9850 | 0.0000 | 0.0000 | 0.0000 | 0.4962 | 0.9188 |
| TabPFNv3 | chb24 | 0.9889 | 0.0000 | 0.0000 | 0.0000 | 0.4972 | 0.5388 |
| TabFM | chb01 | 0.9944 | 0.0000 | 0.0000 | 0.0000 | 0.4986 | 0.7144 |
| TabFM | chb02 | 0.9976 | 0.0000 | 0.0000 | 0.0000 | 0.4994 | 0.8603 |
| TabFM | chb03 | 0.9946 | 0.0000 | 0.0000 | 0.0000 | 0.4986 | 0.8430 |
| TabFM | chb04 | 0.9828 | 0.1333 | 0.4444 | 0.2051 | 0.5982 | 0.8202 |
| TabFM | chb05 | 0.9729 | 0.1562 | 0.5769 | 0.2459 | 0.6160 | 0.9212 |
| TabFM | chb06 | 0.9937 | 0.0000 | 0.0000 | 0.0000 | 0.4984 | 0.7093 |
| TabFM | chb07 | 0.9920 | 0.6364 | 0.4375 | 0.5185 | 0.7572 | 0.8312 |
| TabFM | chb08 | 0.9776 | 0.5000 | 0.0233 | 0.0444 | 0.5166 | 0.6833 |
| TabFM | chb09 | 0.9917 | 0.3750 | 0.2308 | 0.2857 | 0.6408 | 0.8325 |
| TabFM | chb10 | 0.9902 | 0.3333 | 0.0909 | 0.1429 | 0.5690 | 0.9073 |
| TabFM | chb11 | 0.9871 | 0.3333 | 0.0270 | 0.0500 | 0.5217 | 0.4659 |
| TabFM | chb12 | 0.9796 | 0.0000 | 0.0000 | 0.0000 | 0.4949 | 0.7286 |
| TabFM | chb13 | 0.9916 | 0.0000 | 0.0000 | 0.0000 | 0.4979 | 0.6199 |
| TabFM | chb14 | 0.9964 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.6160 |
| TabFM | chb17 | 0.9845 | 0.0667 | 0.0714 | 0.0690 | 0.5306 | 0.7969 |
| TabFM | chb18 | 0.9955 | 0.5000 | 0.1429 | 0.2222 | 0.6100 | 0.9055 |
| TabFM | chb19 | 0.9959 | 0.6667 | 0.1818 | 0.2857 | 0.6418 | 0.8227 |
| TabFM | chb20 | 0.9937 | 0.0000 | 0.0000 | 0.0000 | 0.4984 | 0.8071 |
| TabFM | chb21 | 0.9976 | 1.0000 | 0.1250 | 0.2222 | 0.6105 | 0.8420 |
| TabFM | chb22 | 0.9958 | 0.0000 | 0.0000 | 0.0000 | 0.4989 | 0.8848 |
| TabFM | chb23 | 0.9850 | 0.0000 | 0.0000 | 0.0000 | 0.4962 | 0.9327 |
| TabFM | chb24 | 0.9825 | 0.0000 | 0.0000 | 0.0000 | 0.4956 | 0.6063 |
| Mitra | chb01 | 0.9946 | 0.0000 | 0.0000 | 0.0000 | 0.4987 | 0.6930 |
| Mitra | chb02 | 0.9973 | 0.0000 | 0.0000 | 0.0000 | 0.4993 | 0.6957 |
| Mitra | chb03 | 0.9952 | 1.0000 | 0.0556 | 0.1053 | 0.5514 | 0.8987 |
| Mitra | chb04 | 0.9909 | 0.2222 | 0.3333 | 0.2667 | 0.6310 | 0.7630 |
| Mitra | chb05 | 0.9844 | 0.1538 | 0.2308 | 0.1846 | 0.5884 | 0.6251 |
| Mitra | chb06 | 0.9910 | 0.0000 | 0.0000 | 0.0000 | 0.4977 | 0.6027 |
| Mitra | chb07 | 0.9914 | 0.6667 | 0.2500 | 0.3636 | 0.6796 | 0.7987 |
| Mitra | chb08 | 0.9761 | 0.2000 | 0.0233 | 0.0417 | 0.5148 | 0.5704 |
| Mitra | chb09 | 0.9939 | 0.6250 | 0.3846 | 0.4762 | 0.7366 | 0.8133 |
| Mitra | chb10 | 0.9911 | 0.5000 | 0.0909 | 0.1538 | 0.5747 | 0.8723 |
| Mitra | chb11 | 0.9843 | 0.0000 | 0.0000 | 0.0000 | 0.4961 | 0.2971 |
| Mitra | chb12 | 0.9784 | 0.0000 | 0.0000 | 0.0000 | 0.4945 | 0.7341 |
| Mitra | chb13 | 0.9913 | 0.0000 | 0.0000 | 0.0000 | 0.4978 | 0.6099 |
| Mitra | chb14 | 0.9956 | 0.0000 | 0.0000 | 0.0000 | 0.4989 | 0.6062 |
| Mitra | chb17 | 0.9856 | 0.0000 | 0.0000 | 0.0000 | 0.4964 | 0.7452 |
| Mitra | chb18 | 0.9916 | 0.1667 | 0.2143 | 0.1875 | 0.5916 | 0.8675 |
| Mitra | chb19 | 0.9921 | 0.2778 | 0.4545 | 0.3448 | 0.6704 | 0.8103 |
| Mitra | chb20 | 0.9941 | 0.3333 | 0.1538 | 0.2105 | 0.6038 | 0.6611 |
| Mitra | chb21 | 0.9965 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.7432 |
| Mitra | chb22 | 0.9911 | 0.1176 | 0.2000 | 0.1481 | 0.5718 | 0.8610 |
| Mitra | chb23 | 0.9842 | 0.3333 | 0.0500 | 0.0870 | 0.5395 | 0.8641 |
| Mitra | chb24 | 0.9850 | 0.2222 | 0.1600 | 0.1860 | 0.5892 | 0.5648 |

## Aggregated metrics (mean ± std across folds)

| Model | Folds | Accuracy | Precision | Recall | F1 | F1 Macro | ROC AUC |
|---|---|---|---|---|---|---|---|
| TabICL | 22 | 0.9922 ± 0.0053 | 0.2826 ± 0.4205 | 0.0400 ± 0.0671 | 0.0676 ± 0.1112 | 0.5318 ± 0.0558 | 0.7621 ± 0.1358 |
| TabPFNv3 | 22 | 0.9921 ± 0.0053 | 0.1273 ± 0.3222 | 0.0116 ± 0.0343 | 0.0205 ± 0.0588 | 0.5082 ± 0.0295 | 0.7546 ± 0.1472 |
| TabFM | 22 | 0.9897 ± 0.0069 | 0.2137 ± 0.2824 | 0.1069 ± 0.1661 | 0.1042 ± 0.1389 | 0.5495 ± 0.0695 | 0.7796 ± 0.1193 |
| Mitra | 22 | 0.9898 ± 0.0056 | 0.2190 ± 0.2640 | 0.1182 ± 0.1387 | 0.1253 ± 0.1378 | 0.5601 ± 0.0692 | 0.7135 ± 0.1383 |

## Confusion totals (summed across folds)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| TabICL | 56633 | 11 | 400 | 16 |
| TabPFNv3 | 56638 | 6 | 410 | 6 |
| TabFM | 56440 | 204 | 373 | 43 |
| Mitra | 56465 | 179 | 374 | 42 |

## Most frequently selected features (top 25)

Counted across 22 LOSO folds — how many folds' `SelectKBest` kept each tsfresh feature. Meant as a starting point for feature-importance / XAI analysis, not a substitute for it.

| Rank | Feature | Selected in |
|---|---|---|
| 1 | P7-O1__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var" | 22/22 |
| 2 | P7-O1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max" | 22/22 |
| 3 | P7-O1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min" | 22/22 |
| 4 | P7-O1__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var" | 22/22 |
| 5 | P7-O1__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min" | 22/22 |
| 6 | FP1-F3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max" | 22/22 |
| 7 | FP1-F3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min" | 22/22 |
| 8 | FP1-F3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var" | 22/22 |
| 9 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max" | 22/22 |
| 10 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min" | 22/22 |
| 11 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var" | 22/22 |
| 12 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max" | 22/22 |
| 13 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min" | 22/22 |
| 14 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var" | 22/22 |
| 15 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max" | 22/22 |
| 16 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min" | 22/22 |
| 17 | F3-C3__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var" | 22/22 |
| 18 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max" | 22/22 |
| 19 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min" | 22/22 |
| 20 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var" | 22/22 |
| 21 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max" | 22/22 |
| 22 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min" | 22/22 |
| 23 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var" | 22/22 |
| 24 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max" | 22/22 |
| 25 | C3-P3__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min" | 22/22 |

## Artifacts

- Per-fold metrics CSV: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/loso_fold_metrics.csv`
- TabICL confusion matrix: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/graphs/confusion_tabicl.png`
- TabPFNv3 confusion matrix: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/graphs/confusion_tabpfnv3.png`
- TabFM confusion matrix: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/graphs/confusion_tabfm.png`
- Mitra confusion matrix: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/graphs/confusion_mitra.png`
- ROC curves: `images/results/loso_tabular_chbmit_efficient_ov0.25_fixed/graphs/roc_curves.png`
