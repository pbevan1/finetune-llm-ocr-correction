# llama-3.1-8b-ocr-correction

See [pbevan11/llama-3.1-8b-ocr-correction](https://huggingface.co/pbevan11/llama-3.1-8b-ocr-correction) for model weights.

This model is a fine-tuned version of [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) on the [pbevan11/synthetic-ocr-correction-gpt4o](https://huggingface.co/datasets/pbevan11/synthetic-ocr-correction-gpt4o) dataset.



## Usage

First, download the model 

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model_id='pbevan11/llama-3.1-8b-ocr-correction'
model = AutoPeftModelForCausalLM.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
```

Then, construct the prompt template like so:

```python
def prompt(instruction, inp):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
"""

def prompt_tok(instruction, inp, return_ids=False):
    _p = prompt(instruction, inp)
    input_ids = tokenizer(_p, return_tensors="pt", truncation=True).input_ids.cuda()
    out_ids = model.generate(input_ids=input_ids, max_new_tokens=5000, 
                          do_sample=False)
    ids = out_ids.detach().cpu().numpy()
    if return_ids: return out_ids
    
    full_output = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    response_start = full_output.find("### Response:")
    if response_start != -1:
        return full_output[response_start + len("### Response:"):]
    else:
        return full_output[len(_p):]
```

Finally, you can get predictions like this:

```python
# model inputs
instruction = "You are an assistant that takes a piece of text that has been corrupted during OCR digitisation, and produce a corrected version of the same text."
inp = "Do Not Kule Oi't hy.er-l'rieed AjijqIi: imac - Analyst (fteuiers) Hcuiers - A | ) | ilf, <;/) in |) nter |iic . conic! deeiilf. l.o sell n lower-|)rieofl wersinn oi its Macintosh cornutor to nttinct ronsnnu-rs already euami'red ot its iPod music jiayo-r untl annoyoil. by sccnrit.y problems ivitJi Willtlows PCs , Piper.iaffray analyst. (Jcne Muster <aid on Tlinrtiday."

# print prediction
out = prompt_tok(instruction, inp)
print(out.replace('\\', ' '))
```

This will give you a prediction that looks like this:

  ```md
"Do Not Rule Out Lower-Priced Mac - Analyst (Reuters) Reuters - Apple Inc.  may be considering a lower-priced version of its Macintosh computer to attract consumers already enamored of its iPod music player and annoyed by security problems with Windows PCs, PiperJaffray analyst Gene Munster said on Thursday."
  ```

Alternatively, you can play with this model on Replicate: [https://replicate.com/pbevan1/llama-3.1-8b-ocr-correction](https://replicate.com/pbevan1/llama-3.1-8b-ocr-correction)


## Intended uses & limitations

Reconstructions should not be taken as the truth, the model is likely to make some things up to fill in the gaps, and so some things may not be perfectly histoically acurate.

This model was intended to be used to restore historical documents that have been imperfectly digitalised using OCR.

This model could be used to transform poorly transcribed text into semi-synthetic training data, potentially unlocking millions of tokens of training data for future LLMs. The llama 3.1 license allows training on outputs, so this semi-synthetic data is perfectly legal to use.

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 2
- eval_batch_size: 2
- seed: 49
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 2

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.61          | 0.0331 | 1    | 0.6018          |
| 0.4379        | 0.2645 | 8    | 0.4256          |
| 0.2531        | 0.5289 | 16   | 0.2714          |
| 0.2366        | 0.7934 | 24   | 0.2247          |
| 0.1839        | 1.0331 | 32   | 0.2053          |
| 0.1752        | 1.2975 | 40   | 0.1961          |
| 0.1629        | 1.5620 | 48   | 0.1909          |
| 0.163         | 1.8264 | 56   | 0.1901          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.43.2
- Pytorch 2.1.2+cu118
- Datasets 2.19.1
- Tokenizers 0.19.1

### Citation:
```
@misc {peter_j._bevan_2024,
	author       = { {Peter J. Bevan} },
	title        = { llama-3.1-8b-ocr-correction (Revision 2760c4e) },
	year         = 2024,
	url          = { https://huggingface.co/pbevan11/llama-3.1-8b-ocr-correction },
	doi          = { 10.57967/hf/2791 },
	publisher    = { Hugging Face }
}
```
