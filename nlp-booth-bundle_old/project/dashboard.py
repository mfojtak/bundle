#streamlit run dashboard.py --server.port 4000 --server.address 0.0.0.0 --server.baseUrlPath test
import streamlit as st
import plotly.graph_objects as go
from annotated_text import annotated_text
import json
import SessionState
from transformers.pipelines import Conversation
import pandas as pd
import numpy as np
import tensorflow as tf
import torch as pt

query_params = st.experimental_get_query_params()
app_state = st.experimental_get_query_params()

session_state = SessionState.get(first_query_params=query_params, conversation=Conversation())
first_query_params = session_state.first_query_params

options = ["Welcome", "Sentiment Analysis", "Named Entity Recognition", "Question Answering", 
        "Summarization", "Text Generation", "Conversational AI", "Zero Shot Classification",
        "Masked Language", "Adverse Event Detection", "Machine Translation"] 

default = options.index(first_query_params["option"][0]) if "option" in app_state else 0
option = st.sidebar.radio(
        'Select area of interest',
        options, index=default)

app_state["option"] = option
st.experimental_set_query_params(**app_state)

if option == "Welcome":
    '''
    # Welcome to NLP CoP Booth
    NLP Community of Practice (NLP CoP) is a group for NLP practitioners. The group contains reusable code from gitlab, models and OneNote notebooks for interesting work happening in NLP. It is a place to connect and contribute with like minded NLP geeks.

    Try out some state of art NLP models yourself

    - [Sentiment Analysis](https://computec6.novartis.net/test/?option=Sentiment+Analysis)
    - [Named Entity Recognition](https://computec6.novartis.net/test/?option=Named+Entity+Recognition)
    - [Question Answering](https://computec6.novartis.net/test/?option=Question+Answering)
    - [Summarization](https://computec6.novartis.net/test/?option=Summarization)
    - [Text Generation](https://computec6.novartis.net/test/?option=Text+Generation)
    - [Conversational](https://computec6.novartis.net/test/?option=Conversational)
    - [Zero Shot Classification](https://computec6.novartis.net/test/?option=Zero+Shot+Classification)
    - [Masked Language](https://computec6.novartis.net/test/?option=Masked+Language)
    - [Adverse Event Detection](https://computec6.novartis.net/test/?option=Adverse+Event+Detection)
    - [Machine Translation](https://computec6.novartis.net/test/?option=Machine+Translation)
 
    You can run the model on the pre-filled examples or try out your own text.

    [source](https://gitlab.dev.f1.novartis.net/FOJTAMI1/nlp_booth)
    '''

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification,\
    TFAutoModelForTokenClassification, TFAutoModelForQuestionAnswering, TFAutoModelWithLMHead, \
        TFAutoModelForCausalLM, TFGPT2LMHeadModel, TFT5ForConditionalGeneration, \
        TFRobertaForMaskedLM, AutoModelForTokenClassification
from transformers import pipeline

if option == "Sentiment Analysis":
    '''
    ## Sentiment Analysis
    Sentiment Analysis is the classification of **emotions** as positive or negative within text data. The model below is a multilingual model and uses Bert based model. It generates score between 1-5 with 5 - positive & 1 - negative
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_sentiment_classifier():
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", cache_dir="/data/nlp_booth/cache")
        model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", from_pt=True, cache_dir="/data/nlp_booth/cache")
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        return classifier

    @st.cache
    def get_sentiment(text):
        sentiment_classifier = get_sentiment_classifier()
        return sentiment_classifier(text)

    sentiment_example = st.radio(
            'Select example text (English, Dutch, German, French, Spanish)',
            ("This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
            "Sci-fi manhunt, via Ridley Scott. Formulaic but great-looking. A classic now.",
            "In seiner früheren Fassung war der Film ein Meisterwerk mit Fehlern; in Scotts restaurierter Fassung ist er einfach ein Meisterwerk."))
    sentiment_text = st.text_input('Enter text to analyze', sentiment_example, key="sentiment")
    if sentiment_text:
        res = get_sentiment(sentiment_text)
        i = int(res[0]["label"][0])
        color = "lightgreen"
        if i<3: 
            color = "red"
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = i,
            title = {'text': "Sentiment"},
            gauge = {'axis': {'range': [0, 5]},
                    'bar': {'color': color}}
                    ))
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Did not get what you were expecting?**")
        st.write("The possible reasons could be -")
        st.write("- Out of Domain Text - The text you have typed is out of domain with training data. The model may not give accurate answer when domain changes")
        st.write("- Annotation Challenges - Similar text in the training data could have been annotated differently")
        st.write("- Gray Zone - Model predicts with slightely lower confidence on the correct class")

if option == "Named Entity Recognition":
    '''
    ## Named Entity Recognition(NER) 
    NER seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. The model below is trained on CONLL03 dataset on Google's Electra 
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_ner():
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/electra-large-discriminator-finetuned-conll03-english", cache_dir="/data/nlp_booth/cache")
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/electra-large-discriminator-finetuned-conll03-english", cache_dir="/data/nlp_booth/cache")
        ner = pipeline('ner', model=model, tokenizer=tokenizer)
        return ner, tokenizer
    @st.cache
    def get_entities(text):
        ner, tokenizer = get_ner()
        tokens = tokenizer.tokenize(text)
        entities = ner(text)
        return tokens, entities

    ner_text = st.text_input('Enter text to analyze', 'Ousted WeWork founder Adam Neumann lists his Manhattan penthouse for $37.5 million.', key="ner")
    if ner_text:
        tokens, res = get_entities(ner_text)
        entities = {}
        for word in res:
            entities[word["index"]] = word["entity"]
        mapping = {"I-PER": "#8ef", "I-LOC": "#faa", "I-ORG": "#afa", "I-MISC": "#aff"}
        annotated = []
        for i, token in enumerate(tokens):
            if i+1 in entities:
                entity = entities[i+1]
                anno = (token, entity, mapping[entity])
                annotated.append(anno)
            else:
                annotated.append(token + " ")
        annotated_text(*annotated)
        st.write("**Did not get what you were expecting?**")
        st.write("The possible reasons could be -")
        st.write("- Expectations of labels - If you expect different labels from results then the training data should be different ")
        st.write("- Out of Domain Text - The text you have typed is out of domain with training data. The model may not give accurate answer when domain changes")
        st.write("- Poor Performance - These models can be trained to match human expectations with Reinforcement Learning and Human in the Loop (HILL)")






if option == "Question Answering":
    '''
    ## Question Answering (QA)
    QA is task to find answers to questions posed by humans on a given text or large body of text. The answers can be found in a paragraph (Comprehension based QA) or present in a large collection of data (Open Domain QA). The answers can extracted (Extractive QA) or can be rephrased (Abstractive QA). The model below is example of Comprehension based Extractive QA system trained on BERT on SQUAD Dataset'''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_qa():
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir="/data/nlp_booth/cache")
        model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir="/data/nlp_booth/cache")
        qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=1)
        return qa
    @st.cache
    def get_answer(context, question):
        qa = get_qa()
        return qa(question=question, context=context)

    example1 = {"context": r"""Coronaviruses are a group of RNA viruses that cause diseases in mammals and birds. In humans and birds, they cause respiratory tract infections that can range from mild to lethal. Mild illnesses in humans include some cases of the common cold (which is also caused by other viruses, predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS, and COVID-19.""", \
        "questions": ["What are Coronavirus lethal types?", "What are diseases caused by Coronavirus?"]}
    example2 = {"context": r"""There are three major types of rock: igneous, sedimentary, and metamorphic. The rock cycle is an important concept in geology which illustrates the relationships between these three types of rock, and magma. When a rock crystallizes from melt (magma and/or lava), it is an igneous rock. This rock can be weathered and eroded, and then redeposited and lithified into a sedimentary rock, or be turned into a metamorphic rock due to heat and pressure that change the mineral content of the rock which gives it a characteristic fabric. The sedimentary rock can then be subsequently turned into a metamorphic rock due to heat and pressure and is then weathered, eroded, deposited, and lithified, ultimately becoming a sedimentary rock. Sedimentary rock may also be re-eroded and redeposited, and metamorphic rock may also undergo additional metamorphism. All three types of rocks may be re-melted; when this happens, a new magma is formed, from which an igneous rock may once again crystallize.""", \
        "questions": ["When the three types of rock are re-melted what is formed?", "What changes the mineral content of a rock"]}

    examples = {"Coronavirus Wiki": example1, "Geography": example2}

    qa_context = st.radio(
            'Select example context',
            list(examples.keys()))
    qa_context_txt = st.text_area('Context to analyze', examples[qa_context]["context"])
    qa_question = st.radio(
            'Select example question',
            examples[qa_context]["questions"])
    qa_question_txt = st.text_input('Question', qa_question)
    if qa_question_txt:
        answer = get_answer(qa_context_txt, qa_question_txt)
        st.write("**Answer: **" + answer["answer"])
        st.write("**Did not get what you were expecting?**")
        st.write("The possible reasons could be -")
        st.write("- Extractive vs Abstractive - Did you expect that model will rephrase and create an new answer. This model is trained for Extractive QA dataset hence it selects the answers from text")
        st.write("- Out of domain text - The text you have given as input is not in domain with training data. The model should be trained on your domain to get performance")
        st.write("- Commonsense question? - AI has still long way to go to answer common sense questions. Stay tuned as AI is getting better everyday")
        st.write("- Want to search on large database? - You need **open domain question answering**! The open domain model is trained in a way that it is able to search quickly on a very large body of text which is already converted to vectors")
        st.write("- Answer spread in multiple paragraphs? - You need **multi-hop question answering** which can connect multiple ideas in different paragraph and generate one answer or multiple answers")
        st.write("- Trying to fool the model - Some models come with **no-answer** outcome also!")




if option == "Summarization":
    '''
    ## Summarization
    Summarization, as the name suggests, is the process of shortening of set of data (text, video or audio), to create a subset (a summary) that represents the most important or relevant information within the original content. 
    In machine learning, Summarization can be 
    - Abstractive - AI Model build an internal semantic representation of the original content, and then use this representation to create a summary that is closer to what a human might express
    - Extractive - AI Model indentifies and extracts content from the original data, but the extracted content is not modified in any way.

    Summarization has many applications such as keyphrase extraction, query based summarization, single document summarization, multi-document summarization etc. Training of summarization systems can be done using supervised and unsupervised algorithms

    The model below is a single document abstractive summarization model trained on T5 from Google


    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_sum_model():
        tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="/data/nlp_booth/cache")
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base", cache_dir="/data/nlp_booth/cache")
        sum = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return sum
    @st.cache
    def get_sum(text, length):
        sum_model = get_sum_model()
        summary=sum_model(text, min_length=10, max_length=length)
        return summary

    sum_examples = {"COVID-19 News": "More than 1 million people have died from the coronavirus worldwide, marking another milestone in the pandemic's brief but devastating history. The death toll from the coronavirus, which causes Covid-19, now stands at 1,000,555, according to data from Johns Hopkins University.",
                "Coronavirus Wiki": "Coronaviruses are a group of RNA viruses that cause diseases in mammals and birds. In humans and birds, they cause respiratory tract infections that can range from mild to lethal. Mild illnesses in humans include some cases of the common cold (which is also caused by other viruses, predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS, and COVID-19.",
                "News": "Apple’s iPhone 12 Pro is promising some serious photography capabilities over both its predecessor and the standard iPhone 12. And with that upgrade comes the debut of Apple ProRAW.  This new imaging format has been designed to tap into the upgraded triple rear camera array on both the iPhone 12 Pro and iPhone 12 Pro Max, which also feature a LiDAR sensor to gather more image information.Smartphones have been able to shoot in the RAW format for a while now, but this custom Apple format could usher in a new way to shoot photos rich in data while also tapping into Apple's powerful image processing. "
    }
    sum_example_key = st.radio(
            'Select example text to summarize',
            list(sum_examples.keys()))
    sum_txt = st.text_area('Text to analyze', sum_examples[sum_example_key])
    sum_length = st.slider('Summary length', min_value=10, max_value=100, value=30, step=2)
    sum_model = get_sum_model()
    summary=get_sum(sum_txt, length=sum_length)
    st.write("**Summary: **" + summary[0]["summary_text"])
    st.write("**Did not get what you were expecting?**")
    st.write("Possible reasons could be:-")
    st.write("- Language in-accuracies - Model has been trained on base model. Larger model might solve the problem")
    st.write("- Summary not correct - RL based methods can help the model focus on the right content")
    st.write("- Summary spread in multiple documents - Multi-document summarization algorithm can help you")
    st.write("- Focus on summarizing particular aspect - Query-based summarization will help you")



if option == "Text Generation":
    '''
    ## Text Generation
    Text Generation is a process of converting structured data to language understood by humans. Structured Data can prompt text or keywords or even a table containing relevant information.

    It has several uses in various industries like News, Pharma, Tourism, Retail etc. There has been news about humans finding hard to detect difference between generated and written news. 
    
    Try out text generation for yourself on GPT-2 small model. The biggest GPT-2 model will require 8 GPUs.

    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_gen_model():
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="/data/nlp_booth/cache")
        model = TFGPT2LMHeadModel.from_pretrained("gpt2", cache_dir="/data/nlp_booth/cache")
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return gen

    def generate(text):
        gen_model = get_gen_model()
        res = gen_model(text)
        return res
    gen_examples = ["People are moving to Mars", "Legolas and Gimli attended"]
    gen_example = st.radio(
            'Choose the beggining of your story', gen_examples)
    gen_txt = st.text_input('Enter prompt text', gen_example)
    button = st.button("Generate")
    if button:
        generated_text = generate(gen_txt)
        st.write("**Generated text: **" + generated_text[0]["generated_text"])
        st.write("**Did not get what you were expecting?**")
        st.write("Possible reasons could be:-")
        st.write("- Random sampling - The model generates output by sampling tokens from various probability distributions. It is a random process but output can be guided towards desired goals")
        st.write("- Language in-accuracies - Model has been trained on base model. Larger model might solve the problem")
        st.write("- Missing common sense - Modern Large-scale models are better in commonsense. There are other ways to ensure commonsense using Graph Neural Networks or Knowledge Banks")        
        st.write("- Not what you want? - Conditional Generation Language Models can generate text based on input text, style, or content given as input")



if option == "Conversational AI":
    '''
    ## Conversational AI
    Conversational AI ,Chatbot or Dialogue system is an AI tool to have an real-time chat conversation with AI model. There are generally three types of AI system.
    - Question Answering Chatbot - AI needs to provide concise, direct answers to user queries based on rich knowledge drawn from various data sources including text collections such as Web documents and pre-compiled knowledge bases such as sales and marketing datasets,
    - Task-oriented AI - AI needs to accomplish user tasks ranging from restaurant reservation to meeting scheduling (e.g., Turns 6 to 7 in Fig. 1.1), and to business trip planning.
    - Social Chatbot - AI needs to converse seamlessly and appropriately with users like a human. It can be used for general purpose conversation or guide the humans for their questions 

    The model below is Social Chatbot trained on **DialoGPT** on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 through 2017. You can have more than one message with the Chatbot. 
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_con_model():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir="/data/nlp_booth/cache")
        model = TFAutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", cache_dir="/data/nlp_booth/cache")
        con = pipeline("conversational", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return con

    con_model = get_con_model()
    con_txt = st.text_input('Talk to your bot')
    if con_txt:
        conversation = session_state.conversation
        conversation.add_user_input(con_txt)
        con_model(conversation)
        for user_input, generated_response in zip(conversation.past_user_inputs, conversation.generated_responses):
            st.write("User: " + user_input)
            st.write("**Bot: " + generated_response + "**")
        session_state.conversation = conversation



if option == "Zero Shot Classification":
    '''
    ## Zero Shot Classification
    using **roberta-large-mnli**

    Zero-shot learning (ZSL) is a problem setup in machine learning, where at test time, a learner observes samples from classes that were not observed during training, and needs to predict the category they belong to.
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_zero_model():
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", cache_dir="/data/nlp_booth/cache")
        model = TFAutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", cache_dir="/data/nlp_booth/cache")
        zero = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return zero
    
    @st.cache
    def zero_classify(template, labels, texts):
        zero_model = get_zero_model()
        res = zero_model(texts, labels, template)
        return res
    
    hypothesis_template = st.text_input("Hypothesis template to be combined with class labels",
                "This example is {}.")
    class_labels = st.text_input("Comma separated list of class labels", "politics, public health, sport")
    sequences_txt = st.text_area("Newline separated list of samples to be classified", "Who are you voting for in 2020?\nMore than 1 million people have died from the coronavirus worldwide.\nHarry Kane scored a hat-trick to help Tottenham reach the group stage of the Europa League.")
    sequences = sequences_txt.split("\n")
    zero_result = zero_classify(hypothesis_template, class_labels, sequences)
    for item in zero_result:
        st.write(item["sequence"])
        frame = pd.DataFrame({"label": item["labels"], "score": item["scores"]})
        frame = frame.set_index(["label"])
        st.bar_chart(frame)
    



if option == "Masked Language":
    '''
    ## Masked Language prediction
    using **distilroberta**

    Masked language modeling is an example of autoencoding language modeling (the output is reconstructed from corrupted input) - we typically mask one or more of words in a sentence and have the model predict those masked words given the other words in sentence. By training the model with such an objective, it can essentially learn certain (but not all) statistical properties of word sequences.

    Enter text with one or more <mask> tokens and the model will generate the most likely substitutions.
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_masked_model():
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir="/data/nlp_booth/cache")
        model = TFRobertaForMaskedLM.from_pretrained("distilroberta-base", cache_dir="/data/nlp_booth/cache")
        masked = pipeline("fill-mask", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return masked
    
    @st.cache
    def get_masked(text):
        model = get_masked_model()
        res = model(text)
        return res
    
    masked_examples = ["More than 1 million people have died from the coronavirus worldwide, marking another milestone in the pandemic's brief but devastating history. The death toll from the <mask>, which causes Covid-19, now stands at 1,000,555, according to data from Johns Hopkins University.",
                "Coronaviruses are a group of RNA viruses that cause diseases in <mask> and birds. In humans and birds, they cause respiratory tract infections that can range from mild to lethal. Mild illnesses in humans include some cases of the common cold (which is also caused by other viruses, predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS, and COVID-19."
    ]
    masked_example = st.radio(
            'Choose example with masked tokens', masked_examples)
    masked_txt = st.text_input('Enter masked text', masked_example)
    res = get_masked(masked_txt)
    labels = [item['token_str'][1:] for item in res]
    scores = [item['score'] for item in res]
    st.write("**Proposed substitutions:**")
    frame = pd.DataFrame({"label": labels, "score": scores})
    frame = frame.set_index(["label"])
    st.bar_chart(frame)    



if option == "Adverse Event Detection":
    '''
    ## Adverse Event Detection
    using **xlm-roberta-base**

     Adverse Event Detection is the classification of text as AE event or Non-AE event. The model below is a multilingual model and uses Roberta based model architecture. 
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_adverse_event_classifier():
        output_dir = "/data/nlp_booth/cache/baseline_3class_xlmr_base/"
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework="pt", return_all_scores=True)
        return classifier

    @st.cache
    def get_adverse_event(text):
        adverse_event_classifier = get_adverse_event_classifier()
        return adverse_event_classifier(text)

    adverse_event_example = st.radio(
            'Select example text (English, Dutch, German, French, Spanish)',
            ("Burning in chest, Bitter or sour taste in the back of the throat.",
            "@Novartis Asistiré al stand de One Novartis Data Science NLP booth",
            "Ich habe eine Panikattacke",))

    adverse_event_text = st.text_input('Enter text to analyze', adverse_event_example, key="adverse_event")
    if adverse_event_text:
        res = get_adverse_event(adverse_event_text) 
        frame = pd.DataFrame.from_dict(res[0])
        new_frame = pd.DataFrame(frame[1:2])
        ae_val = frame["score"][0]+frame["score"][2]
        new_frame["label"] = new_frame["label"].str.replace("LABEL_","").replace("1","Non-AE")
        new_frame = new_frame.append({"label":"AE", "score":ae_val}, ignore_index=True)
        new_frame = new_frame.set_index(["label"])
        
        value = np.argmax(np.array(new_frame["score"]))
        
        st.bar_chart(new_frame)
        if value==0:
            st.write("The text is a Non-Adverse Event with probability of {}".format(new_frame['score'][value]))
        else:
            st.write("The text is a Adverse Event with probability of {}".format(new_frame['score'][value]))
        
        st.write("**Did not get what you were expecting?**")
        st.write("The possible reasons could be -")
        st.write("- Out of Domain Text - The text you have typed is out of domain with training data. The model may not give accurate answer when domain changes")
        st.write("- Annotation Challenges - Similar text in the training data could have been annotated differently")
        st.write("- Gray Zone - Model predicts with slightely lower confidence on the correct class")


if option == "Machine Translation":
    '''
    ## Machine Translation
    using **T5**

    Machine translation, sometimes referred to by the abbreviation MT, is a sub-field of computational linguistics that uses AI to translate text or speech from one language to another.
    
    Select one of the 3 sample texts given below or enter your desired text which needs to be translated in the input box and select the language to be translated into. You check the translated version below it.
    '''
    trans_txt_example = st.radio(
            'Select example text to translate from English',
            ("I love One Novartis Data Science Conference",
             "Please check the other ways in which NLP can help you in the sidebar",
             "I am in the NLP booth at the One Novartis Data Science Conference",))

    trans_txt = st.text_input('Enter text to translate', trans_txt_example, key="trans_text")
    
    lang_select = st.radio(
            'Select the language the text needs to be translated into from English',
            ("French", "German", "Romanian",))
    
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_trans_model(lang):
        tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="/data/nlp_booth/cache")
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base", cache_dir="/data/nlp_booth/cache")
        if lang == "French":
            trans = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer, framework="tf", device=1)
        elif lang == "German":
            trans = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer, framework="tf", device=1)
        elif lang == "Romanian":
            trans = pipeline("translation_en_to_ro", model=model, tokenizer=tokenizer, framework="tf", device=1)
        return trans

    @st.cache
    def get_trans_event(text, lang):
        trans_event_translater = get_trans_model(lang)
        return trans_event_translater(text)
    
    if trans_txt:
        res = get_trans_event(trans_txt, lang_select) 
        st.write("The translated text is: **{}**".format(res[0]["translation_text"]))

    st.write("**Did not get what you were expecting?**")
    st.write("Machine Translation is an active area of research. Some of the possible reasons could be -")
    st.write("- Using industry jargon - The translation engine has been not been trained on industry specific domain. The model may not give accurate translations when the domain changes.")
    st.write("- Usage of ambiguous sentences - The translation engine would not give good translations for ambiguous sentences. Ex. Sarah gave a bath to her dog wearing a pink t-shirt. ")
    st.write("- Using slangs - The translation engine may not perform well for language-specific slangs.")
