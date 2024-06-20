"""## Library Imports & Model Loading"""

from flair.models import SequenceTagger
from flair.data import Sentence

from flask import Flask, jsonify
from flask import request

# import inflect

"""Load the pre-trained Flair NER (ontonotes-large) model"""

# tagger = SequenceTagger.load('ner')
# tagger = SequenceTagger.load('ner-large')
tagger = SequenceTagger.load('ner-ontonotes-large')

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch, os

"""## Fine-Tuned RoBERTa for ERR"""

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # 2 labels for binary classification

hf_token = os.getenv('HF_TOKEN')
url="udaykumar97/OASIS_RoBERTa_for_ERR_one"
tokenizer = RobertaTokenizer.from_pretrained(url, token=hf_token)
model = RobertaForSequenceClassification.from_pretrained(url, token=hf_token)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the selected device
model.to(device)

def classify_entities(input_text):
    if '[SEP]' not in input_text:
        # raise ValueError("Input must be formatted with [SEP] to separate parts. Don't include spaces")
        return "Input must be formatted with [SEP] to separate parts. Don't include spaces"

    if 'Which one of the following entities is non-redundant and worth retaining? [SEP]' not in input_text:
        input_text = 'Which one of the following entities is non-redundant and worth retaining?[SEP]' + input_text
    else:
        # raise ValueError("Please don't add the question in the input. It will be added automatically.")
        return "Please don't add the question in the input. It will be added automatically."

    trailing_option = "[SEP]neither are worth retaining[SEP]both are worth retaining"
    if trailing_option not in input_text:
        input_text = input_text + trailing_option
    else:
        # raise ValueError("Please don't add the 'neither' option in the input. It will be added automatically.")
        return "Please don't add the 'neither' option in the input. It will be added automatically."

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform classification
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)

    # Determine the predicted class (0, 1, or 2)
    prediction = torch.argmax(probs, dim=1).item()

    # Split the input text to extract entities
    parts = input_text.split('[SEP]')
    # Adjust the indexing based on your model's prediction
    result = parts[prediction + 1]  # +1 because index 0 is the question

    return result

"""## Taxonomy Assignment Module (TAM) Code Block"""

def programatic_taxonomy_detection(text, taxonomy_list):
    results = []
    for term in taxonomy_list:
        start = 0
        while True:
            start = text.find(term, start)
            if start == -1:
                break
            end = start + len(term)
            results.append({"Text": term, "Type": taxonomy_list[0], "BeginOffset": start, "EndOffset": end})
            start += len(term)  # Move start index beyond the current word to avoid overlapping matches
    return results

custom_taxonomy_name = ['startup', 'business incubator', 'co-founder', 'Liberty Mutual Fire Insurance Company', 'equity crowdfunding']
custom_startup_taxonomy = ['startup', 'business', 'private equity', 'exit strategy', 'burn rate', 'team building', 'performance review', 'valuation cap', 'edtech', 'early-stage startup', 'customer segmentation', 'startup pitch', 'skills training', 'incorporation', 'customer acquisition', 'customer retention', 'trademark registration', 'business plan', 'ipo', 'marketing strategy', 'co-working space', 'market fit', 'venture funding', 'social media strategy', 'angel investor', 'user experience (ux)', 'pivot', 'business accelerator', 'fintech', 'non-disclosure agreement (nda)', 'product launch', 'patent filing', 'leadership', 'equity', 'startup valuation', 'mezzanine financing', 'startup accelerator', 'business model', 'corporation', 'infrastructure as a service (iaas)', 'break-even point', 'late-stage startup', 'proptech', 'software as a service (saas)', 'market disruption', 'run rate', 'market analysis', 'networking events', 'regulatory compliance', 'cap table', 'due diligence', 'management', 'capital raise', 'startup mentor', 'series a/b/c funding', 'scaling', 'minimal viable product (mvp)', 'venture capitalist', 'user acquisition', 'revenue model', 'series c funding', 'term sheet', 'entrepreneurship', 'startup law', 'startup office', 'startup ecosystem', 'series d funding', 'prototype', 'patent', 'venture capital', 'series b funding', 'startup advisor', 'advisor', 'profit margin', 'family and friends round', 'remote work', 'funding round', 'accelerator', 'value proposition', 'board member', 'sales strategy', 'equity stake', 'debt financing', 'platform as a service (paas)', 'startup incubator', 'product development', 'tech stack', 'y combinator', 'proof of concept', 'pitch deck', 'vesting schedule', 'disruptive technology', 'limited partner', 'mvp (minimum viable product)', 'startup competition', 'series a funding', 'mvp', 'bootstrapping', 'staff development', 'venture partner', 'angel round', 'employee hiring', 'sweat equity', 'venture round', 'shareholder agreement', 'seed capital', 'financial model', 'incubator', 'contract negotiation', 'healthtech', 'data privacy', 'pitch competition', 'entrepreneur', 'convertible note', 'innovation', 'growth capital', 'pre-seed funding', 'crowdfunding', 'shares', 'seed round', 'stock options', 'unique selling proposition (usp)', 'competitive analysis', 'traction', 'limited liability company (llc)', 'talent acquisition', 'intellectual property (ip)', 'revenue-based financing', 'startup culture', 'equity financing', 'digital marketing', 'industry conference', 'employment agreement', 'fundraising', 'corporate governance', 'seed funding', 'growth hacking', 'venture debt', 'trademark', 'angel funding', 'partnership', 'business development', 'founder', 'minimum viable product', 'series e funding', 'serial entrepreneur', 'general partner', 'demo day', 'intellectual property rights', 'startup community', 'lean startup', 'go-to-market strategy', 'organizational culture', 'techstars', 'greentech', 'user interface (ui)', 'equity crowdfunding', 'engineer', 'scientist', 'investor']

"""## Flair Code Block"""

def for_doccano_pre_tagging(single_line_paragraph):
  sentence = Sentence(single_line_paragraph)
  tagger.predict(sentence)
  entities = []
  for entity in sentence.get_spans('ner'):
    entities.append([entity.start_position, entity.end_position, entity.labels[0].value])

  results = programatic_taxonomy_detection(single_line_paragraph, custom_startup_taxonomy)
  for result in results:
    entities.append([result['BeginOffset'], result['EndOffset'], result['Type']])

  response = {"text": single_line_paragraph, "label": entities}
  return response

def standardize_entity_text(entity_text, entity_type):
  # p = inflect.engine()
  
  # reduce both entity_text and entity_type to lowercase
  entity_text = entity_text.lower()
  entity_type = entity_type.lower()

  # # if entity_text is plural, convert it to singular
  # if p.singular_noun(entity_text):
  #   entity_text = p.singular_noun(entity_text)

  # remove any leading or trailing whitespaces
  entity_text = entity_text.strip()

  # remove any leading or trailing single or double quotes
  entity_text = entity_text.strip("'")
  entity_text = entity_text.strip('"')

  # remove any occurance of single or double quotes
  entity_text = entity_text.replace("'", "")
  entity_text = entity_text.replace('"', '')

  return entity_text, entity_type

def for_ingestion_pipeline(single_line_paragraph):
  entities, entity_text_only, entity_type_only = [], [], []
  cannonical_map = {}

  # get entities from Vanilla Flair (ner-ontonotes-large model)
  sentence = Sentence(single_line_paragraph)
  tagger.predict(sentence)
  for entity in sentence.get_spans('ner'):
    entity_text_temp = entity.text
    entity_type_temp = entity.labels[0].value

    # if entity_text_temp is 'cardinal' or 'ordinal', skip it
    if entity_type_temp == 'CARDINAL' or entity_type_temp == 'ORDINAL':
      continue

    entity_text_temp, entity_type_temp = standardize_entity_text(entity_text_temp, entity_type_temp)

    entity_text_with_type = entity_text_temp + " (" + entity_type_temp + ")"
    # entity_text_with_type = entity_text_with_type.replace("'", "") # replace all occurances of single and double quotes
    entities.append(entity_text_with_type)

    if entity_text_temp not in entity_text_only:
      entity_text_only.append(entity_text_temp)

    if entity_type_temp not in entity_type_only:
      entity_type_only.append(entity_type_temp)

    if entity_type_temp in cannonical_map:
      if entity_text_temp not in cannonical_map[entity_type_temp]:
        cannonical_map[entity_type_temp].append(entity_text_temp)
    else:
      cannonical_map[entity_type_temp] = [entity_text_temp]

  results = programatic_taxonomy_detection(single_line_paragraph, custom_startup_taxonomy)
  for result in results:
    entity_text_temp = result['Text']
    entity_type_temp = result['Type']

    entity_text_temp, entity_type_temp = standardize_entity_text(entity_text_temp, entity_type_temp)

    entity_text_with_type = entity_text_temp + " (" + entity_type_temp + ")"
    entities.append(entity_text_with_type)

    if entity_text_temp not in entity_text_only:
      entity_text_only.append(entity_text_temp)

    if entity_type_temp not in entity_type_only:
      entity_type_only.append(entity_type_temp)

    if entity_type_temp in cannonical_map:
      if entity_text_temp not in cannonical_map[entity_type_temp]:
        cannonical_map[entity_type_temp].append(entity_text_temp)
    else:
      cannonical_map[entity_type_temp] = [entity_text_temp]

  entities = list(set(entities))
  response = {"Entities": entities, "EntityTextOnly": entity_text_only, "EntityTypeOnly": entity_type_only, "CannonicalMap": cannonical_map}
  return response

"""## Flask Code Block"""

app = Flask(__name__)

@app.route('/', methods=["POST"])
def hello():
  data = request.get_json()
  text = data['text']
  recognized_entities = {"warning": "This is not an active/production endpoint. Production enpoints include 'ingestion_pipeline' and 'doccano_pre_annotation'"}
  recognized_entities = jsonify(recognized_entities) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response
  return recognized_entities

@app.route('/doccano_pre_annotation', methods=["POST"])
def doccano_pre_annotation():
  data = request.get_json()
  text = data['text']
  recognized_entities = for_doccano_pre_tagging(text) # for doccano pre-annotation of text
  recognized_entities = jsonify(recognized_entities) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response

  # debug prints start
  print("Input Text in body:", text)
  print("Recognized Entities Object:", recognized_entities)
  print("Recognized Entities Object Type:", type(recognized_entities))
  print("Recognized Entities JSON Contents:", recognized_entities.json)
  print("\n\n")
  # debug prints end
  return recognized_entities

@app.route('/ingestion_pipeline', methods=["POST"])
def ingestion_pipeline():
  data = request.get_json()
  text = data['text']
  recognized_entities = for_ingestion_pipeline(text) # for ingestion pipeline
  recognized_entities = jsonify(recognized_entities) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response

  # debug prints start
  print("\nInput Text in body:", text)
  # print("Recognized Entities Object:", recognized_entities)
  # print("Recognized Entities Object Type:", type(recognized_entities))
  print("Recognized Entities JSON Contents:", recognized_entities.json)
  # print("\n\n")
  # debug prints end
  return recognized_entities

@app.route('/err', methods=["POST"])
def err():
  data = request.get_json()
  text = data['text']
  non_redundant_entity = classify_entities(text)
  non_redundant_entity = jsonify(non_redundant_entity) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response
  return non_redundant_entity