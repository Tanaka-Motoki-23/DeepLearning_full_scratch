# DeepLearning_full_scratch
Javaを用いたニューラルネットワークのモデル実装・学習アルゴリズムのフルスクラッチ実装です。

# ファイル構成
```
/src
├── Data.java
├── Dataset.java
├── LinerArray.java
├── Layer.java
├── Affine.java
├── ReLU.java
├── Sigmoid.java
├── SoftmaxWithLoss.java
├── ActivateFunctions.java
├── Optimizer.java
├── SGD.java
├── Momentum.java
├── AdaGrad.java
├── NetworkWithBackPropagation.java
├── MnistNet.java
└── Main.java
```

## プログラム概要
入力データと正解ラベルをまとめたクラス
```
Data.java
```
複数のデータを読み込み管理するためのクラス
```
Dataset.java
```
行列計算やブロードキャストを利用して処理を記述するためのクラス
```
LinerArray.java
```
ネットワークのレイヤー実装の基礎となるインターフェース
```
Layer.java
```
全結合層のレイヤー実装
```
Affine.java
```
ReLUのレイヤー実装
```
ReLU.java
```
Sigmoid関数のレイヤー実装
```
Sigmoid.java
```
Softmax関数とCrossEntropy関数を合わせたレイヤー実装
```
SoftmaxWithLoss.java
```
様々な活性化関数の基礎実装
```
ActivateFunctions.java
```
勾配降下法に利用するオプティマイザー実装の基礎となるインターフェース
```
Optimizer.java
```
SGDの実装
```
SGD.java
```
Momentum SGDの実装
```
Momentum.java
```
AdaGradの実装
```
AdaGrad.java
```
ニューラルネットワーク作成の基礎となるクラス
```
NetworkWithBackPropagation.java
```
MNIST(input=784)に対応したネットワーク実装
```
MnistNet.java
```
モデルの学習・テストのサンプルコード
```
main.java
```
