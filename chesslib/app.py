from flair.models import SequenceTagger
from flair.data import Sentence

from flask import Flask, jsonify
from flask import request

import inflect
p = inflect.engine()

# Load the pre-trained NER model
# tagger = SequenceTagger.load('ner')
# tagger = SequenceTagger.load('ner-large')
tagger = SequenceTagger.load('ner-ontonotes-large')

def programatic_taxonomy_detection(text, taxonomy_list):
  # convert all text to lowercase
  text = text.lower()
  results = []
  word_boundaries = {' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '\n', '\t'}
  for term in taxonomy_list:
      start = 0
      while True:
          start = text.find(term.lower(), start)
          if start == -1: # if no more occurrences are found
              break

          end = start + len(term)

          # Check if the term is a whole word
          if (start == 0 or text[start - 1] in word_boundaries) and (end == len(text) or text[end] in word_boundaries):
              results.append({"Text": term, "Type": taxonomy_list[0], "BeginOffset": start, "EndOffset": end})
            
          start += len(term)  # Move start index beyond the current word to avoid overlapping matches
  return results

custom_startup_taxonomy = ['startup', 'business', 'private equity', 'exit strategy', 'burn rate', 'team building', 'performance review', 'valuation cap', 'edtech', 'early-stage startup', 'customer segmentation', 'startup pitch', 'skills training', 'incorporation', 'customer acquisition', 'customer retention', 'trademark registration', 'business plan', 'ipo', 'marketing strategy', 'co-working space', 'market fit', 'venture funding', 'social media strategy', 'angel investor', 'user experience (ux)', 'pivot', 'business accelerator', 'fintech', 'non-disclosure agreement (nda)', 'product launch', 'patent filing', 'leadership', 'equity', 'startup valuation', 'mezzanine financing', 'startup accelerator', 'business model', 'corporation', 'infrastructure as a service (iaas)', 'break-even point', 'late-stage startup', 'proptech', 'software as a service (saas)', 'market disruption', 'run rate', 'market analysis', 'networking events', 'regulatory compliance', 'cap table', 'due diligence', 'management', 'capital raise', 'startup mentor', 'series a/b/c funding', 'scaling', 'minimal viable product (mvp)', 'venture capitalist', 'user acquisition', 'revenue model', 'series c funding', 'term sheet', 'entrepreneurship', 'startup law', 'startup office', 'startup ecosystem', 'series d funding', 'prototype', 'patent', 'venture capital', 'series b funding', 'startup advisor', 'advisor', 'profit margin', 'family and friends round', 'remote work', 'funding round', 'accelerator', 'value proposition', 'board member', 'sales strategy', 'equity stake', 'debt financing', 'platform as a service (paas)', 'startup incubator', 'product development', 'tech stack', 'y combinator', 'proof of concept', 'pitch deck', 'vesting schedule', 'disruptive technology', 'limited partner', 'mvp (minimum viable product)', 'startup competition', 'series a funding', 'mvp', 'bootstrapping', 'staff development', 'venture partner', 'angel round', 'employee hiring', 'sweat equity', 'venture round', 'shareholder agreement', 'seed capital', 'financial model', 'incubator', 'contract negotiation', 'healthtech', 'data privacy', 'pitch competition', 'entrepreneur', 'convertible note', 'innovation', 'growth capital', 'pre-seed funding', 'crowdfunding', 'shares', 'seed round', 'stock options', 'unique selling proposition (usp)', 'competitive analysis', 'traction', 'limited liability company (llc)', 'talent acquisition', 'intellectual property (ip)', 'revenue-based financing', 'startup culture', 'equity financing', 'digital marketing', 'industry conference', 'employment agreement', 'fundraising', 'corporate governance', 'seed funding', 'growth hacking', 'venture debt', 'trademark', 'angel funding', 'partnership', 'business development', 'founder', 'minimum viable product', 'series e funding', 'serial entrepreneur', 'general partner', 'demo day', 'intellectual property rights', 'startup community', 'lean startup', 'go-to-market strategy', 'organizational culture', 'techstars', 'greentech', 'user interface (ui)', 'equity crowdfunding', 'VCs', ]

def get_entities_from_paragraphs(paragraphs):
  entities = []
  for paragraph in paragraphs:
    sentence = Sentence(paragraph)
    tagger.predict(sentence)
    entities.extend([(entity.text, entity.labels[0].value) for entity in sentence.get_spans('ner')])
  return entities

def get_entities_from_paragraph(paragraph):
  sentence = Sentence(paragraph)
  tagger.predict(sentence)
  entities = []
  for entity in sentence.get_spans('ner'):
    entities.append({"Text": entity.text, "Type": entity.labels[0].value, "BeginOffset": entity.start_position, "EndOffset": entity.end_position})
  response = {"Entities": entities}
  return response

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

def for_ingestion_pipeline(single_line_paragraph):
  sentence = Sentence(single_line_paragraph)
  tagger.predict(sentence)
  entities, entity_text_only, entity_type_only = [], [], []
  cannonical_map = {}
  for entity in sentence.get_spans('ner'):
    entity_text_temp = entity.text
    entity_type_temp = entity.labels[0].value

    # if entity_text_temp is plural, convert it to singular
    if p.singular_noun(entity_text_temp):
      entity_text_temp = p.singular_noun(entity_text_temp)

    # remove any leading or trailing whitespaces
    entity_text_temp = entity_text_temp.strip()

    # remove any leading or trailing single or double quotes
    entity_text_temp = entity_text_temp.strip("'")
    entity_text_temp = entity_text_temp.strip('"')

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

    # if entity_text_temp is plural, convert it to singular
    if p.singular_noun(entity_text_temp):
      entity_text_temp = p.singular_noun(entity_text_temp)

    # remove any leading or trailing whitespaces
    entity_text_temp = entity_text_temp.strip()

    # remove any leading or trailing single or double quotes
    entity_text_temp = entity_text_temp.strip("'")
    entity_text_temp = entity_text_temp.strip('"')

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

  entities = list(set(entities))
  response = {"Entities": entities, "EntityTextOnly": entity_text_only, "EntityTypeOnly": entity_type_only, "CannonicalMap": cannonical_map}
  return response

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

if __name__ == '__main__':
    app.run(port=5000)

