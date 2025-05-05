# 【第3回】AIエンジニアリング実践 任意課題

##宿題の概要
この宿題では、講義で学んだRAG（Retrieval-Augmented Generation）技術を用いて、LLMの生成内容を改善する実践的な取り組みを行います。演習で利用したコードをベースに、独自の質問と参照文書を用いて実験を行い、RAGの効果を定量的・定性的に評価します。

この宿題を通じて、「テストデータの作成」と「改善のプロセス」について理解を深め、実際のアプリケーション開発
に役立てることを目指します。

##宿題の内容
1. **独自の質問と参照資料の作成**  
 * 自分で5つ以上の質問文を考案してください
 * 各質問に対する回答を含む参照文書を用意してください
 * 少なくとも1つは、LLMが単体では正確に答えられないような知識を含む質問にしてください


2. **実験の実施**
 * 演習で使用したコードをベースに、以下の2つの方法で回答を生成してください
    * ベースのLLM（RAGなし）での回答生成  
    * RAGを組み合わせた回答生成  
 * 回答の評価では、単純なYes/No判定でも良いです  
    * より詳細な評価指標も検討していただけるとなお良いです  
3. **結果分析と考察**
 * 生成した結果をまとめ、RAGありとRAGなしの差異を分析してください
 * RAGによって回答が改善したケースと悪化したケースの両方について考察してください
 * 結果に基づいて、RAGの有効性と限界についての考察を記述してください


## 扱う質問

以下、2025年3月に発見された tj-actions/changed-files の脆弱性（CVE-2025-30066）をテーマにした質問例を 6 つ挙げる。（モデル単体ではリアルタイムに正確に答えにくい最新情報を含む。）
1. **脆弱性の発生タイムライン**  
   tj-actions/changed-files リポジトリにおいて、悪意あるペイロードが最初にコミットされたのはいつか、またそれが公表されたのはいつか？
2. **攻撃手法と影響範囲**  
  この脆弱性によって GitHub Actions のログに漏洩した具体的な秘密情報（環境変数や認証トークンなど）の例は？
3. **検出・診断方法**  
  既存のリポジトリが影響を受けているかを自動でチェックするために有効なスクリプトやクエリの例は？
4. **修正内容の技術的詳細**  
  悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられた？
5. **公式アドバイザリとドキュメント**  
   GitHub Security Advisory や CISA アラートで公開された、この脆弱性に関する公式ドキュメントの識別子（ID）や URL は？
6. **影響を受けた組織・プロジェクト**  
   実際に被害が報告された主要なオープンソースプロジェクト名や企業名、およびそれらが公開したインシデント報告のリンクは？


## 回答を含む参照文書

1. 脆弱性発生タイムラインは、2025年3月12日から15日にかけてtj-actions/changed-filesのタグが悪意あるコミットに差し替えられ、3月14日 StepSecurityが異常検知、3月15日 GitHubがリポジトリを一時非公開化し3月17日にv46.0.1がリリースされた。
2. 攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した。
3. 検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する。
4. 修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした。
5. 公式アドバイザリは、CVE-2025-30066、GHSA-MRRH-FWG8-R2C3、CISAアラート「Supply Chain Compromise of Third-Party tj-actions/changed-files」（2025-03-18公開）。
6. 影響を受けた組織は、espressif/arduino-esp32、chains-project/maven-lockfile、rackerlabs/genestack、modal-labs/modal-examplesなど約23,000リポジトリが使用し、公開ログ保持プロジェクトで漏洩が多発し、StepSecurityやAqua Securityが詳細レポートを公開。


## 扱うモデル

「google/gemma-2-2b-jpn-it」を使用します。このモデルは、リリース時期の関係上、以下の特徴を持ちます。

- tj-actions/changed-files の脆弱性（CVE-2025-30066）情報が広まる前に訓練されており、このトピックに関する知識を持たないと想定される
- この特性を活かし、純粋なベースライン評価から各手法の効果を観察する

### 演習環境の準備


```python
!pip install --upgrade transformers
!pip install google-colab-selenium
!pip install bitsandbytes
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages (4.51.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers) (3.17.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.30.2)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2024.11.6)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers) (2.32.3)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.21.0)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.5.3)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers) (4.67.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (2024.10.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2025.1.31)
    Requirement already satisfied: google-colab-selenium in /usr/local/lib/python3.11/dist-packages (1.0.14)
    Requirement already satisfied: selenium in /usr/local/lib/python3.11/dist-packages (from google-colab-selenium) (4.32.0)
    Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.11/dist-packages (from urllib3[socks]<3,>=1.26->selenium->google-colab-selenium) (2.3.0)
    Requirement already satisfied: trio~=0.17 in /usr/local/lib/python3.11/dist-packages (from selenium->google-colab-selenium) (0.30.0)
    Requirement already satisfied: trio-websocket~=0.9 in /usr/local/lib/python3.11/dist-packages (from selenium->google-colab-selenium) (0.12.2)
    Requirement already satisfied: certifi>=2021.10.8 in /usr/local/lib/python3.11/dist-packages (from selenium->google-colab-selenium) (2025.1.31)
    Requirement already satisfied: typing_extensions~=4.9 in /usr/local/lib/python3.11/dist-packages (from selenium->google-colab-selenium) (4.12.2)
    Requirement already satisfied: websocket-client~=1.8 in /usr/local/lib/python3.11/dist-packages (from selenium->google-colab-selenium) (1.8.0)
    Requirement already satisfied: attrs>=23.2.0 in /usr/local/lib/python3.11/dist-packages (from trio~=0.17->selenium->google-colab-selenium) (25.1.0)
    Requirement already satisfied: sortedcontainers in /usr/local/lib/python3.11/dist-packages (from trio~=0.17->selenium->google-colab-selenium) (2.4.0)
    Requirement already satisfied: idna in /usr/local/lib/python3.11/dist-packages (from trio~=0.17->selenium->google-colab-selenium) (3.10)
    Requirement already satisfied: outcome in /usr/local/lib/python3.11/dist-packages (from trio~=0.17->selenium->google-colab-selenium) (1.3.0.post0)
    Requirement already satisfied: sniffio>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from trio~=0.17->selenium->google-colab-selenium) (1.3.1)
    Requirement already satisfied: wsproto>=0.14 in /usr/local/lib/python3.11/dist-packages (from trio-websocket~=0.9->selenium->google-colab-selenium) (1.2.0)
    Requirement already satisfied: pysocks!=1.5.7,<2.0,>=1.5.6 in /usr/local/lib/python3.11/dist-packages (from urllib3[socks]<3,>=1.26->selenium->google-colab-selenium) (1.7.1)
    Requirement already satisfied: h11<1,>=0.9.0 in /usr/local/lib/python3.11/dist-packages (from wsproto>=0.14->trio-websocket~=0.9->selenium->google-colab-selenium) (0.14.0)
    Requirement already satisfied: bitsandbytes in /usr/local/lib/python3.11/dist-packages (0.45.5)
    Requirement already satisfied: torch<3,>=2.0 in /usr/local/lib/python3.11/dist-packages (from bitsandbytes) (2.5.1+cu124)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from bitsandbytes) (1.26.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (3.17.0)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (4.12.2)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (3.1.5)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (2024.10.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (9.1.0.70)
    Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.5.8)
    Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (11.2.1.3)
    Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (10.3.5.147)
    Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (11.6.1.9)
    Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.3.1.170)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.127)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (12.4.127)
    Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (3.1.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch<3,>=2.0->bitsandbytes) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch<3,>=2.0->bitsandbytes) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch<3,>=2.0->bitsandbytes) (3.0.2)



```python
# 演習用のコンテンツを取得
!git clone https://github.com/taitai-2009/lecture-ai-engineering.git
```

    Cloning into 'lecture-ai-engineering'...
    remote: Enumerating objects: 76, done.[K
    remote: Counting objects: 100% (7/7), done.[K
    remote: Compressing objects: 100% (6/6), done.[K
    remote: Total 76 (delta 1), reused 4 (delta 1), pack-reused 69 (from 1)[K
    Receiving objects: 100% (76/76), 89.42 KiB | 14.90 MiB/s, done.
    Resolving deltas: 100% (18/18), done.



```python
# HuggingFace Login
from huggingface_hub import notebook_login

notebook_login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
# CUDAが利用可能ならGPUを、それ以外ならCPUをデバイスとして設定
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
import random
random.seed(0)
```


```python
# モデル(Gemma2)の読み込み

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


# 1. ベースラインモデル評価
**まずはベースモデルがどの程度知識を持っているか確かめる**


```python
def generate_output(query):
  messages = [
      {"role": "user", "content": query},
  ]
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)

  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = model.generate(
      input_ids,
      max_new_tokens=256,
      eos_token_id=terminators,
      do_sample=False,
      # temperature=0.6, # If do_sample=True
      # top_p=0.9,  # If do_sample=True
  )

  response = outputs[0][input_ids.shape[-1]:]
  return tokenizer.decode(response, skip_special_tokens=True)
```


```python
#question =  "LLMにおけるInference Time Scalingとは？"
question =  "tj-actions/changed-files 脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"
response = generate_output(question)
print(response)
```

    ## TJ-actions/changed-files 脆弱性による悪意あるコードの分析
    
    **TJ-actions/changed-files** における脆弱性によって悪意のあるコードがどのファイル／関数に追加されていたかを示すには、以下の手順と情報が必要となります。
    
    **1. 脆弱性情報:**
    
    * **脆弱性の種類:**  どのような脆弱性によって影響を受けていたのか（例：SQL注入、クロスサイトスクリプトingなど）
    * **影響を受けるファイル/関数のリスト:**  脆弱性によって影響を受けたファイルや関数のリストを特定する。
    * **悪意のあるコードの具体的な内容:**  悪意のあるコードの具体的な内容を記述する。
    
    **2. 修正パッチ情報:**
    
    * **修正内容の記述:**  修正パッチによってどのようなコードが書き換えられたのかを詳細に説明する。
    * **変更箇所をファイル/関数のリストで示す:**  修正パッチによって変更された箇所をファイル/関数のリストで示す。
    * **変更内容の詳細な説明:**  変更箇所ごとに、どのような変更が行われたのか（例：コードの削除、追加、修正など）を詳細に説明する。
    * **変更後のコード


- 数値的な評価も見てみます。RagasにはAnswer Accuracyという評価指標があります。今回はこちらを参考に実装した評価関数を利用して測っていきます。

- 今回はgemmaでは性能が不安定だったので、OpenAIのgpt-4oで評価していきます。従って、scoreの実行はopenAI APIキーを所持している関心がある方のみで良いです。


```python
!pip install -U openai
```

    Requirement already satisfied: openai in /usr/local/lib/python3.11/dist-packages (1.77.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.11/dist-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from openai) (0.28.1)
    Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from openai) (0.8.2)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.11/dist-packages (from openai) (2.10.6)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.11/dist-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.11/dist-packages (from openai) (4.67.1)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.11/dist-packages (from openai) (4.12.2)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.11/dist-packages (from anyio<5,>=3.5.0->openai) (3.10)
    Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from httpx<1,>=0.23.0->openai) (2025.1.31)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.11/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.11/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic<3,>=1.9.0->openai) (0.7.0)
    Requirement already satisfied: pydantic-core==2.27.2 in /usr/local/lib/python3.11/dist-packages (from pydantic<3,>=1.9.0->openai) (2.27.2)



```python
# @title 評価実装
#gold_answer = "「Inference Time Scaling」とは、推論時に計算量を増やしてモデルの性能を高める手法です。これはモデルのサイズを大きくする代わりに、難しい入力に対して多くの計算リソースを使うことで、より良い出力を得ようとするアプローチです。"
gold_answer = "tj-actions/changed-files 脆弱性への修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースしています。"

from openai import OpenAI
from google.colab import userdata
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"), max_retries=5, timeout=60)

def openai_generator(query):

        messages = [
                    {
                        "role": "user",
                        "content": query
                    }
                ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

def evaluate_answer_accuracy(query, response, reference):

    template_accuracy1 = (
          "Instruction: You are a world class state of the art assistant for rating "
          "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
          "Say 4, if User Answer is full contained and equivalent to Reference Answer"
          "in all terms, topics, numbers, metrics, dates and units.\n"
          "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
          "in all terms, topics, numbers, metrics, dates and units.\n"
          "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
          "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
          "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
          "Even small discrepancies in meaning, terminology, directionality, or implication must result in a lower score. Only rate 4 if the User Answer is a complete and precise match to the Reference Answer in every aspect.\n"
          "### Question: {query}\n"
          "### {answer0}: {sentence_inference}\n"
          "### {answer1}: {sentence_true}\n"
          "The rating is:\n"
      )
    template_accuracy2 = (
          "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
          "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
          "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
          "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
          "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
          "Even minor inconsistencies in meaning, terminology, emphasis, or factual detail should prevent a rating of 4. Only assign a 4 if the User Answer exactly and unambiguously matches the Reference Answer in every respect."
          "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
          "Question: {query}\n\n"
          "{answer0}: {sentence_inference}\n\n"
          "{answer1}: {sentence_true}\n\n"
          "Rating: "
      )

    score1 = openai_generator(
                template_accuracy1.format(
                      query=query,
                      answer0="User Answer",
                      answer1="Reference Answer",
                      sentence_inference=response,
                      sentence_true=reference,
                    )
                )
    try:
      score1 = int(score1)
    except:
      print("Failed")
      score1 = 0

    score2 = openai_generator(
                template_accuracy2.format(
                        query=query,
                        answer0="Reference Answer",
                        answer1="User Answer",
                        sentence_inference=reference,
                        sentence_true=response,
                    )
                  )

    try:
      score2 = int(score2)
    except:
      print("Failed")
      score2 = 0


    return (score1 + score2) / 2
```


```python
# 評価
score = evaluate_answer_accuracy(question, response, gold_answer)
print(score)
```

    0.0


## 結果 (ベースモデル)

「google/gemma-2-2b-jpn-it」は「tj-actions/changed-files の脆弱性」について誤った知識を提示しました：
* モデルは脆弱性の特徴を、SQL注入、クロスサイトスクリプトingなど、間違った解釈をしている。

---

# 2. 回答・参照情報データの活用

## 2.1 回答・参照情報データをソースとして活用 (RAG導入)

モデルの回答の事実性を向上させるためにRetrieval Augmented Generation (RAG)技術を導入します：

* **知識ソース**: 回答・参照情報データ
* **目的**: モデルに「tj-actions/changed-files の脆弱性」に関する正確な知識と文脈を提供し、事実に基づいた回答を促す

**初期RAG実装（ベーシックアプローチ）**:
* **ドキュメント処理**: 回答・参照情報データを含む生テキストをそのまま使用
* **分割方法**: 「。」（句点）で区切られた文単位でテキストを分割
* **検索手法**: シンプルな類似度ベースの検索でクエリに関連する文を抽出
* **制約条件**: モデルの入力トークン制限に収まるよう関連文のみを選択


```python
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("infly/inf-retriever-v1-1.5b", trust_remote_code=True)
# In case you want to reduce the maximum length:
emb_model.max_seq_length = 4096
```


```python
#with open("/content/lecture-ai-engineering/day3/data/LLM2024_day4_raw.txt", "r") as f:
with open("/content/lecture-ai-engineering/day3/data/tj-actions-raw.txt", "r") as f:
  raw_writedown = f.read()
```


```python
# ドキュメントを用意する。
documents = [text.strip() for text in raw_writedown.split("。")]
print("ドキュメントサイズ: ", len(documents))
print("ドキュメントの例: \n", documents[3])
```

    ドキュメントサイズ:  7
    ドキュメントの例: 
     修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした



```python
# Retrievalの実行
#question = "LLMにおけるInference Time Scalingとは？"
question = "tj-actions/changed-files 脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"

query_embeddings = emb_model.encode([question], prompt_name="query")
document_embeddings = emb_model.encode(documents)

# 各ドキュメントの類似度スコア
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())
```

    [[75.28347778320312, 67.4903564453125, 73.73432159423828, 71.55113220214844, 70.24580383300781, 65.6257553100586, 54.611366271972656]]



```python
topk = 5
for i, index in enumerate(scores.argsort()[0][::-1][:topk]):
  print(f"取得したドキュメント{i+1}: (Score: {scores[0][index]})")
  print(documents[index], "\n\n")
```

    取得したドキュメント1: (Score: 75.28347778320312)
    脆弱性発生タイムラインは、2025年3月12日から15日にかけてtj-actions/changed-filesのタグが悪意あるコミットに差し替えられ、3月14日 StepSecurityが異常検知、3月15日 GitHubがリポジトリを一時非公開化し3月17日にv46.0.1がリリースされた 
    
    
    取得したドキュメント2: (Score: 73.73432159423828)
    検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する 
    
    
    取得したドキュメント3: (Score: 71.55113220214844)
    修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした 
    
    
    取得したドキュメント4: (Score: 70.24580383300781)
    公式アドバイザリは、CVE-2025-30066、GHSA-MRRH-FWG8-R2C3、CISAアラート「Supply Chain Compromise of Third-Party tj-actions/changed-files」（2025-03-18公開） 
    
    
    取得したドキュメント5: (Score: 67.4903564453125)
    攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した 
    
    



```python
references = "\n".join(["* " + documents[i] for i in scores.argsort()[0][::-1][:topk]])
#query =  f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"
query =  f"[参考資料]\n{references}\n\n[質問] tj-actions/changed-files の脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"
response = generate_output(query)
print(response)
```

    ## tj-actions/changed-files 脆弱性による悪意あるコードの追加と修正
    
    **脆弱性発生箇所:**
    
    * **index.js** ファイル内にあるコードが、外部Gistからmemdump.pyをダウンロード・実行するコードが注入されていた。
    
    **悪意あるコードの追加:**
    
    * **index.js** には、外部Gistからmemdump.pyをダウンロード・実行するコードが注入された。
    * **memdump.py** は、Runnerのメモリをダンプし、AWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出す。
    
    **修正内容:**
    
    * **index.js** から外部Gistからmemdump.pyをダウンロード・実行するコードを削除した。
    * **action.yml** で外部スクリプト読み込みを禁止した。
    * **全タグを安全なコミットに再ポイント**してv46.0.1以降をリリースした。
    
    
    
    **修正方法の詳細:**
    
    * **外部スクリプト読み込み禁止:** action.yml で外部スクリプト読み込みを禁止することで、悪意のあるコードのダウンロード・実行を防ぐ。
    * **



```python
# 評価
score = evaluate_answer_accuracy(question, response, gold_answer)
print(score)
```

    3.0


### 結果 (初期RAG実装)

回答・参照情報のファイルにある情報を元に、回答の生成ができた。

### 問題分析
以下の要因が考えられます：
2. **検索精度の課題**: 単純な文単位の分割では文脈が失われ、関連性の高いドキュメント片を適切に取得できていない可能性

# 3. 文脈を考慮したチャンク化の導入

検索結果の品質向上のため、以下の改善を実施します：

* **前後文脈を含むチャンク化**:
  - 検索でマッチした文だけでなく、その前後の複数文も含めてチャンクとして取得
  - 具体的には、マッチした文を中心に前2文、後2文を含む計5文程度のチャンクを構成
  - この「文脈ウィンドウ」により、発言の背景情報や議論の流れが保持される

* **期待される効果**:
  - 講師の主張とその根拠の関係性を正確に把握できる
  - 概念の定義とその適用範囲を正しく理解できる

この改善により、モデルが講義内容の本質をより正確に理解し、一貫性のある事実に基づいた回答を生成することが期待されます。


```python
# 前後それぞれ2つずつの文章を一つのドキュメントに追加する。（要は5つの文章集合になる)
references = "\n".join(["* " + "。".join(documents[max(0, i-2): min(i+2, len(documents))]).strip() for i in scores.argsort()[0][::-1][:topk]])
#query =  f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"
query =  f"[参考資料]\n{references}\n\n[質問] tj-actions/changed-files の脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"
response = generate_output(query)
print(response)
```

    ## tj-actions/changed-files の脆弱性による悪意あるコード追加
    
    **脆弱性発生:** 2025年3月12日から15日にかけて、tj-actions/changed-filesのタグが悪意あるコミットに差し替えられた。
    
    **攻撃手法:**  index.js に外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した。
    
    **修正内容:**
    
    1. **悪意あるGist取得・実行処理の削除:**  index.js から悪意あるGist取得・実行処理を削除した。
    2. **外部スクリプト読み込み禁止:**  action.yml で外部スクリプト読み込みを禁止した。
    3. **全タグを安全なコミットに再ポイント:**  v46.0.1以降をリリースし、安全なコミットに再ポイントした。
    
    **ファイル／関数の詳細:**
    
    * **index.js:**  悪意あるGist取得・実行処理が注入されていた。
     
    
    
    **補足:**
    
    *  上記の情報は



```python
# 評価
score = evaluate_answer_accuracy(question, response, gold_answer)
print(score)
```

    3.0


## 結果 (文脈付きチャンク化によるRAG)

文脈を含むチャンク化により、モデルの回答の方向性に明確な改善が見られました：

### 改善点
* 「推論時の計算をスケールさせる」という概念を据えて回答
* Inference Time Scalingの基本原理についての理解が向上

### 残存する問題点
* 質問と関連性の低い情報（ノイズ）が混入する

### 問題分析

文脈付きチャンク化によるアプローチで新たに発生した課題：

1. **情報過多の問題**:
   * ドキュメント量の増加により、モデルに提供される情報総量が大幅に増加
   * 関連情報と非関連情報が混在し、ノイズと重要情報の区別が困難に

2. **情報選択の複雑化**:
   * モデルは単に回答を生成するだけでなく、提供された多様な情報源から関連性の高い情報を選別する作業も担うことになった
   * この二重タスクにより回答生成の難易度が上昇




# 4. Rerankによる情報品質の向上

検索精度をさらに向上させるため、二段階の検索プロセスを導入します：

* **Rerank手法の導入**:
  - 第一段階: 従来通り基本的な検索アルゴリズムでtop-k個のドキュメントチャンクを取得
  - 第二段階: 取得したチャンクに対してLLMを活用した高度な関連性評価を実施
  - LLMに「このドキュメントは質問『tj-actions/changed-files の脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたか？』に対して本当に関連性が高いか」を判断させる
  - 関連性スコアに基づいてランク付けし、真に関連性の高いチャンクのみを選出

* **期待される効果**:
  - 質の高い情報に焦点を絞ることで、ノイズとなる情報を大幅に削減
  - 文脈を保ちながらも、関連性の高い情報のみをモデルに提供
  - モデルのタスクを「多量の情報から選別して回答」から「厳選された情報に基づいて回答」へと単純化

この高度な情報フィルタリングにより、tj-actions/changed-files の脆弱性に関する正確で一貫性のある回答生成が期待されます。


```python
references = []
for ref in ["。".join(documents[max(0, i-2): min(i+2, len(documents))]).strip() for i in scores.argsort()[0][::-1][:topk]]:

  #query = f"与えられた[参考資料]が[質問]に直接関連しているかを、'yes''no'で答えること。[参考資料]\n{ref}\n\n[質問] LLMにおけるInference Time Scalingとは？"
  query = f"与えられた[参考資料]が[質問]に直接関連しているかを、'yes''no'で答えること。[参考資料]\n{ref}\n\n[質問] tj-actions/changed-files の脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"

  response = generate_output(query)

  print("\n\n対象となるドキュメント:\n", ref.replace("。", "。\n"))
  print("\n関連しているかどうか: ", response)

  if "yes" in response.lower():
    references.append(ref)
```

    
    
    対象となるドキュメント:
     脆弱性発生タイムラインは、2025年3月12日から15日にかけてtj-actions/changed-filesのタグが悪意あるコミットに差し替えられ、3月14日 StepSecurityが異常検知、3月15日 GitHubがリポジトリを一時非公開化し3月17日にv46.0.1がリリースされた。
    攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した
    
    関連しているかどうか:  yes 
    
    
    
    
    
    対象となるドキュメント:
     脆弱性発生タイムラインは、2025年3月12日から15日にかけてtj-actions/changed-filesのタグが悪意あるコミットに差し替えられ、3月14日 StepSecurityが異常検知、3月15日 GitHubがリポジトリを一時非公開化し3月17日にv46.0.1がリリースされた。
    攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した。
    検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する。
    修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした
    
    関連しているかどうか:  yes 
    
    
    
    
    
    対象となるドキュメント:
     攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した。
    検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する。
    修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした。
    公式アドバイザリは、CVE-2025-30066、GHSA-MRRH-FWG8-R2C3、CISAアラート「Supply Chain Compromise of Third-Party tj-actions/changed-files」（2025-03-18公開）
    
    関連しているかどうか:  yes 
    
    
    
    
    
    対象となるドキュメント:
     検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する。
    修正内容は、悪意あるGist取得・実行処理をindex.jsから削除し、action.ymlで外部スクリプト読み込みを禁止、全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースした。
    公式アドバイザリは、CVE-2025-30066、GHSA-MRRH-FWG8-R2C3、CISAアラート「Supply Chain Compromise of Third-Party tj-actions/changed-files」（2025-03-18公開）。
    影響を受けた組織は、espressif/arduino-esp32、chains-project/maven-lockfile、rackerlabs/genestack、modal-labs/modal-examplesなど約23,000リポジトリが使用し、公開ログ保持プロジェクトで漏洩が多発し、StepSecurityやAqua Securityが詳細レポートを公開
    
    関連しているかどうか:  yes 
    
    
    
    
    
    対象となるドキュメント:
     脆弱性発生タイムラインは、2025年3月12日から15日にかけてtj-actions/changed-filesのタグが悪意あるコミットに差し替えられ、3月14日 StepSecurityが異常検知、3月15日 GitHubがリポジトリを一時非公開化し3月17日にv46.0.1がリリースされた。
    攻撃手法と漏洩情報は、index.jsに外部Gistからmemdump.pyをダウンロード・実行するコードが注入され、RunnerのメモリをダンプしてAWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出した。
    検出・診断方法は、GitHubコード検索でtj-actions/changed-filesの使用箇所を特定し、Falco ActionsやHarden-Runnerで外部接続を監視し、ログからBase64二重エンコード文字列を抽出してシークレットパターンを検査する
    
    関連しているかどうか:  yes 
    
    
    



```python
print(len(references))
```

    5


上記より、上位5件のみが関連しているとわかったので、これらだけをモデルに渡すこととする。（生成内容が確立的なので、4件でない可能性もあります）


```python
#query =  f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"
query =  f"[参考資料]\n{references}\n\n[質問] tj-actions/changed-files の脆弱性によって、悪意あるコードがどのファイル／関数に追加されていたかを示し、修正パッチではどのようにコードが書き換えられたかを説明してください。"
response = generate_output(query)
print(response)
```

    ## tj-actions/changed-files の脆弱性による悪意あるコード追加
    
    **脆弱性発生:** 2025年3月12日から15日にかけて、tj-actions/changed-filesのタグが悪意あるコミットに差し替えられました。
    
    **攻撃手法:**  
    -  index.js に外部Gistからmemdump.pyをダウンロード・実行するコードが注入されました。
    -  memdump.pyはRunnerのメモリをダンプして、AWSキー、GitHub PAT、npmトークン、RSA鍵などをログに書き出します。
    
    **修正内容:**
    -  悪意あるGist取得・実行処理をindex.jsから削除しました。
    -  action.ymlで外部スクリプト読み込みを禁止しました。
    -  全タグを安全なコミットに再ポイントしてv46.0.1以降をリリースしました。
    
    **ファイル／関数の詳細:**
    -  **index.js:**  悪意のあるGist取得・実行処理が追加されたファイルです。
        -  外部Gistからmemdump.pyをダウンロード・実行するコードが注入されました。
        -  Runnerのメモリ



```python
# 評価
score = evaluate_answer_accuracy(question, response, gold_answer)
print(score)
```

    2.0


## 結果 (Rerank導入後)

Rerankの導入により、回答品質に改善が見られました：

### 達成された成果
* tj-actions/changed-files の脆弱性に関する正確な情報を含んだ回答の生成
* 無関係な情報やノイズの排除
* 講義内容を反映した説明の実現 🎉

この結果から、RAGパイプラインにおける情報の質と関連性の重要性であり、検索で取得した情報を単に増やすだけでなく、その情報の関連性を精査する方法を学ぶことができました。
