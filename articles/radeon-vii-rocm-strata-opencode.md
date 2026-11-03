---
title: "Radeon VIIとUbuntu 26.04でROCmを動かし、Flash-NextをOpenCodeにつないだ"
emoji: "🟥"
type: "tech"
topics: [rocm, radeon, localllm, opencode, ubuntu]
published: false
---

5年ほど前に組んだデスクトップがある。当時、GPUにはすでに型落ちだったRadeon VIIを5万円くらいで入手して載せた。Stable Diffusionで遊んでいた時期もあったが、飽きてからはすっかり出番がなくなっていた。

最近はローカルLLMが盛り上がっている。とはいえ、パソコンもGPUもクソ高い。新しいGPUを買う前に、あまり期待せず、久しぶりにこのデスクトップに火を入れて環境を組み直してみることにした。

気になったのは、Radeon VIIのROCm公式サポートが切れていることだ。かなり苦戦する覚悟だったが、実際にはChatGPTのGPT-6.1 Terra（medium）に相談しながら、環境の調査から構築、動作確認まで進められた。多少の修正はあったものの、さくさく進む。最終的に、Qwen3.8-Flash-Nextが動き、OpenCodeからローカルモデルへツール付きでアクセスできるようになった。古いデスクトップでvibe codingを試せる環境ができてしまい、だいぶびっくりしている。

この記事は、そのRadeon VIIでROCmを動かし、ローカルLLMをOpenCodeにつなぐまでの記録です。

結果は、Q2_0量子化版のFlash-Nextで短めの会話の生成が約35 tokens/s。別途、Qwen3.8-27BのQ3量子化版もllama.cppで約21〜22 tokens/s出ています。両者は量子化も実行エンジンも違うため、モデル同士の純粋な比較ではありません。

この記事は2026年11月3日時点の実機メモです。**gfx906のROCmはAMDの通常の動作保証対象とは別扱い**で、以下は一つの環境で動いた記録です。アップデート後も同じように動くとは限りません。

## 環境と構成

| 項目 | 今回の環境 |
| --- | --- |
| GPU | Radeon VII、VRAM 16GB、gfx906 |
| CPU | Ryzen 5 3600、6コア12スレッド、AVX2 |
| RAM | 公称64GB、OSから約60.7GiB |
| OS | Ubuntu 26.04.1 LTS |
| カーネル | 7.0.0-42-generic、標準のamdgpu/KFD |
| ROCm | TheRock nightly `10.3.0a20261102`、gfx906向け |
| HIPのバージョン表示 | `7.18.26434` |
| 推論 | Strata v0.2.10、gfx906用ソースビルド |
| モデル | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF、Q2_0 |
| コーディングUI | OpenCode v1.19.5 |

Flash-Nextは本体125B、1トークンで有効になるのは約6BのMoEモデルで、別に51BのN-gram参照テーブルがあります。Q2_0はパラメーターを削った小型モデルではなく、重みを量子化したものです。[Qwenの説明](https://github.com/QwenLM/Qwen3.8-Flash-Next)と[量子化版の配布元](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)を参照してください。

構成は大きく3層です。

1. Ubuntu標準のamdgpu/KFDがGPUを扱う。
2. gfx906向けROCmのユーザー空間ライブラリとHIPが計算を担当する。
3. StrataがモデルをGPU・RAM・ストレージに配置し、ローカルAPIを公開する。

コンテナは使わず、ROCmやビルド成果物はホームディレクトリ下に分離しました。システムのドライバやAPTリポジトリは変更していません。

## まずamdgpuとKFDを確認する

画面が映っているだけではGPU計算が動くとは限りません。まずカーネルドライバとデバイスノードを確認しました。

```bash
uname -a
lspci -nnk | grep -A 4 -i 'vga\|display\|3d'
ls -l /dev/kfd /dev/dri
getfacl /dev/kfd /dev/dri/renderD128
```

このPCでは`1002:66af`のRadeon VIIを`amdgpu`が使用し、`/dev/kfd`と`renderD128`にはログインユーザー向けのACLが付いていました。アクセス権の設定方法は環境によって異なります。`/dev/dri/renderD128`も固定とは限らないので、自分の環境で確認します。

## gfx906向けROCmをユーザー領域へ

[TheRockの対応表](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)ではgfx906はビルド対象ですが、Sanity Tested・Release Readyにはチェックがありません。配布されていることと、実機で動作検証されていることは別です。

今回はAMDのnightlyから、同じ日付のSDK・ライブラリ・開発ツール・gfx906向けデバイスコードを仮想環境に揃えました。`uv`は別途用意しています。以下の版は実験時点の固定値で、nightlyの保持期間や配布状態によっては取得できなくなる可能性があります。

```bash
uv venv ~/rocm-gfx906/venv
uv pip install --python ~/rocm-gfx906/venv/bin/python \
  --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
  'rocm[libraries,devel,device-gfx906]==10.3.0a20261102'

source ~/rocm-gfx906/venv/bin/activate
rocm-sdk init
export ROCM_PATH="$(rocm-sdk path --root)"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
rocminfo
```

これは配置先を分かりやすく書き直した再現例です。実機では`~/.local/share/rocm-gfx906/venv`を使いました。Ubuntu標準のPythonでは`venv`の`ensurepip`がなかったため、`uv`で作成しています。

`rocminfo`ではRadeon VIIが`gfx906:sramecc+:xnack-`として見えました。さらにHIPで4,096要素のベクトル加算、rocBLASで2×2と512×512の行列積を検証しました。SDK付属テストも27項目が通過しています。

```text
HIP compiled vector addition: PASS (4096 elements)
rocBLAS SGEMM 512x512 identity product: PASS
Ran 27 tests ... OK
```

`amdflang`は最初、`libquadmath.so.0`不足で起動しませんでした。今回は`libquadmath0`を`apt download`してユーザー領域に展開し、ライブラリパスを通して解決しました。同様にC++開発ヘッダーとGCCの開発ライブラリもユーザー領域へ展開しています。一般的な環境では、必要な開発パッケージをディストリビューションの方法で揃える方が簡単です。無関係なUbuntu版のROCmパッケージを混ぜないようにしました。

## 最初にllama.cppでGPU推論を確認

Strataに進む前に、llama.cppをHIP有効・`gfx906`指定でビルドしました。使用したllama.cppのコミットは`f9448cddc48c129a4abc7ff9c2a84c2b4c101c61`です。ビルド方法の基本は[llama.cppの公式ドキュメント](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)にあります。

`llama-cli --list-devices`に`ROCm0: AMD Radeon VII`が表示され、Qwen2.5-0.5BのGGUFで短い生成に成功。次に[Qwen3.8-27BのUD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)も試しました。モデルファイルは約13.2GBです。

```bash
llama-server -m /path/to/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port 8082
```

日本語の短い質問に正常回答し、会話では約21〜22 tokens/s。後にログを詳細化して、66/66層がGPUに配置されていることも確認しました。別の8K設定では、約7,390トークンの合成入力を約240 tokens/sで処理しています。いずれも今回のハードウェアと設定での観測です。

## Strataをgfx906向けにビルドする

[StrataのAMD向けドキュメント](https://github.com/Niko1221/Strata/blob/v0.2.10/docs/AMD_HIP.md)には、gfx906をwave64互換レイヤーでビルドする手順があります。配布済みのRX 6000/7000/9000向けバイナリをRadeon VIIに流用するのではなく、専用の`STRATA_HIP_GFX906`経路を使います。

Strata v0.2.10のソースと、プロジェクトが指定するllama.cppコミット`3cf03257f219afbe7334045ff7c6a06ac68c627d`を用意しました。システムにC/C++開発環境、CMake、Ninja、ROCmの開発ツールが必要です。

```bash
# STRATA_SRCとLLAMA_SRCは、それぞれ取得・展開したソースディレクトリ
cmake -S "$STRATA_SRC" -B "$STRATA_SRC/build-906" -G Ninja \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DCMAKE_C_COMPILER="$ROCM_PATH/bin/amdclang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/amdclang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$STRATA_SRC/build-906" --target strata strata-device -j 6
```

実機ではシステムにGCC開発環境がなかったため、APTからダウンロードして展開した`libstdc++-15-dev`と`libgcc-15-dev`を指す`--gcc-toolchain`をC/C++/HIPの各フラグに追加しています。上は開発環境が通常の場所にある場合の基本形です。

### ビルド時に互換レイヤーの定義を追加した

v0.2.10をそのままビルドすると、`cudaDeviceGetStreamPriorityRange`と`cudaStreamCreateWithPriority`が未定義になりました。gfx906互換ヘッダー`include/strata/platform/hip_compat/strata_hip.h`に、対応するHIP APIへのマッピングを追加して再ビルドしました。

```diff
 #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
+#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
+#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
```

これは今回の版とビルドで必要だったローカル修正です。将来の版では修正済みかもしれません。ビルド後、`strata-device --selftest`、gfx906向けGDN演算テスト、専門家演算の比較テストで失敗がないことを確認しました。これらは全機能や長時間の安定性を保証するものではありません。

## Flash-Nextを準備して起動

モデルは[ISTA-DASLabのGGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)の`Q2_0`を選びました。37.6GBの本体と28.8GBの参照テーブル、合わせて約66.4GBです。今回使用したリビジョンは`ed59f92082b1e93c0e96d60a8b11aab089b52f09`。両ファイルを配布元のSHA-256と照合しました。

```text
Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
SHA256 69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720
Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
SHA256 316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
```

転送が途中で停滞して再始動した際、最初の不完全ファイルを引き継がず、新しいファイルとして再取得しました。完成後に古い不完全ファイル約17.6GBを削除しています。大きなモデルでは、途中再開の動作と空き容量を確認したいところです。

取得後、Strataに付属する`tools/iq_pack.py`でインデックス、dense重み、トークナイザーを準備します。`STRATA_GGUF_PY`は上で用意した固定コミットのllama.cppにある`gguf-py`を指します。

```bash
export STRATA_GGUF_PY="$LLAMA_SRC/gguf-py"
python "$STRATA_SRC/tools/iq_pack.py" \
  --gguf /path/to/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --out /path/to/pack-q2_0
```

サーバー設定には、ビルドした`strata`、pack、2つのGGUF、付属の`data/expert-profile.bin`を指定します。主要なエンジン引数は次のとおりです。

```json
{
  "exe": "/path/to/build-906/strata",
  "cwd": "/path/to/Strata",
  "tokenizer": "/path/to/pack-q2_0/tokenizer",
  "model_name": "Qwen3.8-Flash-Next-Q2_0-gfx906",
  "backend": "hip",
  "gpu": 0,
  "parallel": 1,
  "drafting": "off",
  "args": [
    "--pack", "/path/to/pack-q2_0",
    "--native", "/path/to/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
    "--expert-profile", "/path/to/Strata/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--max-context", "262144",
    "--kv", "int8",
    "--kv-resident", "32768",
    "--ple-io", "direct",
    "--spec", "2"
  ]
}
```

パスは自分の環境に置き換えます。Pythonの依存パッケージは別の仮想環境に`requirements.txt`から導入し、ROCmのライブラリパスも通した状態で、ソースルートからサーバーを起動しました。

```bash
cd "$STRATA_SRC"
python -m serve.server --engine strata --config /path/to/config.json \
  --host 127.0.0.1 --port 8081
```

最初は`--spec`を付け忘れ、ネイティブpackには`--spec T (T >= 2)`が必要だとエラーになりました。`--spec 2`を追加すると読み込みが進み、日本語での応答に成功しました。ここでの`--spec 2`はエンジンの指定で、MTP補助モデルを導入したという意味ではありません。今回`drafting`はoffです。

サーバーはローカルのみにbindしています。稼働確認は以下で行えます。

```bash
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8081/v1/models
```

## 8Kから32K、256Kへ

まず8,192トークンで動作確認し、その後32,768へ増やしました。約30,417トークンの合成入力の中央に置いた合言葉を正しく回収できています。入力処理は約71.8秒、約424 tokens/sでした。

次に`--max-context 262144 --kv int8 --kv-resident 32768`で起動。約248,458トークンの入力でも、中央の合言葉を正しく回収できました。入力処理は約640秒、約388 tokens/sでした。

| 試行 | 結果 |
| --- | --- |
| 8K設定、短い会話 | 約35 tokens/sの生成を観測 |
| 32K設定、約30K入力 | 合言葉を正しく回収、入力約424 tokens/s |
| 256K設定、約248K入力 | 合言葉を正しく回収、入力約388 tokens/s |

長文は同じ日本語の文章を繰り返した合成データで、検索課題も一つです。複雑な複数文書の理解、長時間稼働、異なるコンテンツで同じ精度・速度が出ることまでは確認していません。短い出力6トークンの生成速度を、長文回答の安定速度として扱うこともしません。

256K起動時は専門家キャッシュが8,235個、空きVRAMは約790MiBでした。GPUがデスクトップ表示も担当しているので余裕は小さめです。`--expert-cache auto`やVRAM予約量の調整が必要になる場合があります。[Strataのメモリ設定](https://github.com/Niko1221/Strata/blob/v0.2.10/docs/DETAILS.md)を確認してください。

また、今回のStrataの既定設定では`STRATA_ROUTE_TAIL_SKIP=7`が有効で、キャッシュにない専門家の一部を省略します。ログにも回答が少し変わり得ると表示されます。7,390トークンの合成入力で1回ずつ試したところ、生成は有効時36.7 tokens/s、無効時33.7 tokens/sで、回答文も少し変わりました。各1回の短い応答なので、この差を一般化したり品質差を判断したりはできません。

## OpenCodeにつなぐ

StrataのOpenAI互換APIを使います。OpenCode v1.19.5をユーザー領域に導入し、`~/.config/opencode/opencode.json`にローカルプロバイダーを追加しました。設定形式は[OpenCodeのプロバイダー設定](https://opencode.ai/docs/providers)を参照しています。

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "strata/Qwen3.8-Flash-Next-Q2_0-gfx906",
  "provider": {
    "strata": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strata (local Radeon VII)",
      "options": {
        "baseURL": "http://127.0.0.1:8081/v1",
        "apiKey": "local",
        "timeout": 1200000
      },
      "models": {
        "Qwen3.8-Flash-Next-Q2_0-gfx906": {
          "name": "Qwen3.8 Flash Next Q2_0 (local)",
          "tool_call": true,
          "limit": { "context": 262144, "output": 8192 }
        }
      }
    }
  }
}
```

モデルIDは`/v1/models`が返すIDと一致させます。既存設定がある場合は丸ごと置き換えず、必要なプロバイダーを統合してください。`apiKey`はこのローカルの無認証サーバーに渡すダミー値です。

```bash
opencode models strata
opencode
```

簡単な接続確認として、OpenCodeにテストファイルを読むよう依頼したところ、`read`ツールを呼び出し、ファイル内の合言葉を正しく返しました。コードの変更や複雑なエージェント作業の品質までは、この時点では検証していません。

## 振り返り

Radeon VIIの16GB VRAMだけにモデル全体を載せるのではなく、GPUの専門家キャッシュ、RAM、SSDを組み合わせることで、Flash-Nextの推論までたどり着けました。gfx906は標準的な導入経路から外れているため、GPU認識だけで成功とせず、HIP、行列演算、エンジン、実モデル、ツール呼び出しの順に確かめるのが役立ちました。

一方、今回はnightly ROCm、ソースビルド、互換ヘッダーへの小さな修正を使っています。Qwen3.8-27BはStrataではなくllama.cppで動かしており、Strataで同じ27Bが動くという話ではありません。まずは手元で動いた版と設定を記録し、更新するときは一つずつ比較していこうと思います。

## この記事について

実はこの記事も、実際に環境を構築したChatGPTのセッションで書きました。会話の中で調べたこと、実行したコマンド、ログに残った測定結果をもとにChatGPTが下書きしています。OpenCodeではファイル読み取りツールの呼び出しまで確認しましたが、複雑なコード修正の成功率はまだ測っていません。これから、この環境でどこまでできるか試していきたいと思います。
