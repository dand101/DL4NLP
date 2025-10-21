import torch
import string
from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, \
    logging

logging.set_verbosity_error()

no_words_to_be_predicted = globals()
select_model = globals()
enter_input_text = globals()


def set_model_config(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

    no_words_to_be_predicted = list(kwargs.values())[0]
    select_model = list(kwargs.values())[1]
    enter_input_text = list(kwargs.values())[2]

    return no_words_to_be_predicted, select_model, enter_input_text


def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
            return bert_tokenizer, bert_model
        elif model_name.lower() == "gpt":
            gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            gpt_model = AutoModelWithLMHead.from_pretrained("gpt2")
            return gpt_tokenizer, gpt_model
        else:
            xlnet_tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
            xlnet_model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")
            return xlnet_tokenizer, xlnet_model
    except Exception as e:
        pass


# bert encode
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


# bert decode
def decode_bert(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


# gpt encode
def encode_gpt(tokenizer, text_sentence, add_special_tokens=False):
    input_ids = tokenizer.encode(text_sentence, return_tensors="pt")
    return input_ids


# gpt decode
def decode_gpt(tokenizer, model, input_ids, top_clean=5):
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + top_clean,
            do_sample=True,
            top_k=top_clean,   # top-k filtering
            top_p=1.0,         # nucleus sampling threshold
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded



# xlnet encode
def encode_xlnet(tokenizer, text_sentence):
    PADDING_TEXT = "animal or thing <eod> </s> <eos>"
    input_ids = tokenizer.encode(PADDING_TEXT + text_sentence, add_special_tokens=False, return_tensors="pt")
    return input_ids


def decode_xlnet(text_sentence, tokenizer, pred, prompt_length):
    resulting_string = text_sentence + tokenizer.decode(pred[0])[prompt_length:]
    print(resulting_string)


def get_all_predictions(text_sentence, model_name, top_clean=5):
    if model_name.lower() == "bert":
        # ========================= BERT =================================
        print(no_words_to_be_predicted)
        input_ids, mask_idx = encode_bert(bert_tokenizer, text_sentence)
        with torch.no_grad():
            predict = bert_model(input_ids)[0]
        bert = decode_bert(bert_tokenizer, predict[0, mask_idx, :].topk(no_words_to_be_predicted).indices.tolist(),
                           top_clean)
        return {'bert': bert}

    elif model_name.lower() == "gpt":
        # ========================= GPT =================================
        input_ids = encode_gpt(gpt_tokenizer, text_sentence)
        with torch.no_grad():
            outputs = gpt_model.generate(input_ids, max_length=input_ids.shape[1] + no_words_to_be_predicted,
                                         do_sample=True, top_k=50, top_p=0.95,
                                         num_return_sequences=1, pad_token_id=gpt_tokenizer.eos_token_id, )
        decoded_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        gpt = decoded_text[len(enter_input_text):].strip()
        return {'gpt': gpt}

    else:
        # ========================= XLNet =================================
        input_ids = encode_xlnet(xlnet_tokenizer, text_sentence)

        with torch.no_grad():
            prompt_length = input_ids.shape[1] + no_words_to_be_predicted
            print(prompt_length)
            predict = xlnet_model.generate(input_ids, max_length=prompt_length, do_sample=True, top_p=0.95,
                                           top_k=top_clean)

        xlnet = xlnet_tokenizer.decode(predict[0])[prompt_length:]
        for special in ["<eod>", "<eos>", "</s>"]:
            xlnet = xlnet.replace(special, "")
        # xlnet = decode_xlnet(text_sentence, xlnet_tokenizer, predict, prompt_length)
        return {'xlnet': xlnet}


def get_prediction_end_of_sentence(input_text, model_name):
    try:
        if model_name.lower() == "bert":
            input_text += ' <mask>'
            print(input_text)
            res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
            return res
        elif model_name.lower() == "gpt":
            print(input_text)
            res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
            return res
        else:
            print(input_text)
            res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
            return res

    except Exception as error:
        pass


try:
    print("Next Word Prediction with Pytorch using BERT, GPT, and XLNet")
    for model_name in ['bert', 'gpt', 'xlnet']:
        print("==========", model_name,"=============")
        no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=2,
                                                                                    select_model=model_name,
                                                                                 enter_input_text="I am waiting for")
        print()
        if select_model.lower() == "bert":
            no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=1,
                                                                                        select_model=model_name,
                                                                                        enter_input_text="I am waiting for")
            bert_tokenizer, bert_model = load_model(select_model)
            res = get_prediction_end_of_sentence(enter_input_text, select_model)
            print("result is: {}".format(res))
            answer_bert = []
            print(res['bert'].split("\n"))
            for i in res['bert'].split("\n"):
                answer_bert.append(i)
                answer_as_string_bert = "    ".join(answer_bert)
                print("output answer is: {}".format(answer_as_string_bert))
                # print(f"Predicted List is Here: {answer_as_string_bert}")
            for i in res['bert'].split("\n"):
                input = enter_input_text +" " +i
                no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=1,
                                                                                            select_model=model_name,
                                                                                            enter_input_text=input)

                res = get_prediction_end_of_sentence(enter_input_text, select_model)
                print("result is: {}".format(res))
                # print("output answer is: {}".format(answer_as_string_bert))


        elif select_model.lower() == "gpt":
            gpt_tokenizer, gpt_model = load_model(select_model)
            continuation = get_prediction_end_of_sentence(enter_input_text, select_model)
            print("Result is: {}".format(continuation))

        else:
            xlnet_tokenizer, xlnet_model = load_model(select_model)
            res = get_prediction_end_of_sentence(enter_input_text, select_model)
            print("result is: {}".format(res))


except Exception as e:
    print(e)
    print('Some problem occurred')