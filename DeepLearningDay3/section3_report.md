# Section3: GRU

## 1. 要点まとめ





<div style="page-break-before:always"></div>

-----
## 2. 実装演習

1_3_stochastic_gradient_descent.ipynbの「確率勾配降下法」のコードをベースに、数値微分による学習コードを作成し、誤差逆伝播との実行時間の差を確認する。

``` python


``` 

<div style="page-break-before:always"></div>

実行結果は以下。

| バイト長 | SimpleRNN | LSTM |
|:-----------|:-----------|:------------|
| 8　　　 | loss=0.000014 <br/><img src="section2_fig_kadai_res_simple_bin8.png" width="50%" /> | loss=0.0026 <br/><img src="section2_fig_kadai_res_lstm_bin8.png" width="50%" /> |
| 16　　　 | loss=0.000015 <br/><img src="section2_fig_kadai_res_simple_bin16.png" width="50%" /> | loss=0.0013 <br/><img src="section2_fig_kadai_res_lstm_bin16.png" width="50%" /> |
| 32　　　 | loss=0.000006 <br/><img src="section2_fig_kadai_res_simple_bin32.png" width="50%" /> | loss=0.0011 <br/><img src="section2_fig_kadai_res_lstm_bin32.png" width="50%" /> |

<img src="section5_kadai_result.png" width="75%" />

<div style="page-break-before:always"></div>

-----
## 3. 確認テスト

<img src="section1_test1.png" width="75%" />


<div style="page-break-before:always"></div>

-----

<img src="section1_test2.png" width="75%" />

<div style="page-break-before:always"></div>

-----

<img src="section1_test3.png" width="75%" />

<div style="page-break-before:always"></div>

-----

<img src="section1_test4.png" width="75%" />

<div style="page-break-before:always"></div>

-----

<img src="section1_test5.png" width="75%" />

