"""## Library Imports & Model Loading"""

from flair.models import SequenceTagger
from flair.data import Sentence

from flask import Flask, jsonify
from flask import request

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import json, os, time, re
from fuzzywuzzy import fuzz
from tqdm import tqdm

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

entity_blacklist = ["of", "the", "a", "an", "and", "or", "but", "so", "for", "to", "in", "on", "at", "by", "with", "from", "as", "into", "onto", "upon", "after", "before", "since", "during", "while", "about", "against", "between", "among", "through", "over", "under", "above", "below", "behind", "beside", "beneath", "around", "near", "off", "out", "up", "down", "away", "back", "forward", "around", "throughout", "within", "without", "inside", "outside", "between", "beyond", ",", "its"]

def classify_entities_and_return_parameters_batch(input_text):
    # for each line in the input_text, classify the entities and return the parameters
    results = []
    for line in input_text:
      if '[SEP]' not in line:
          type_of_input = type(input_text)
          type_of_line = type(line)
          return f"Input must be formatted with [SEP] to separate parts. Don't include spaces. \nType of input: {type_of_input}\nline: {line}\nType of line: {type_of_line}"

      if 'Which one of the following entities is non-redundant and worth retaining? [SEP]' not in line:
          line = 'Which one of the following entities is non-redundant and worth retaining?[SEP]' + line
      else:
          return "Please don't add the question in the input. It will be added automatically."

      trailing_option = "[SEP]neither are worth retaining[SEP]both are worth retaining"
      if trailing_option not in line:
          line = line + trailing_option
      else:
          return "Please don't add the 'neither' option in the input. It will be added automatically."

      # Tokenize the input text
      inputs = tokenizer(line, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
      inputs = {k: v.to(device) for k, v in inputs.items()}

      # Perform classification
      outputs = model(**inputs)
      logits = outputs.logits

      # Convert logits to probabilities using softmax
      probs = torch.softmax(logits, dim=1)

      # Determine the predicted class (0, 1, or 2)
      prediction = torch.argmax(probs, dim=1).item()

      # Split the input text to extract entities
      parts = line.split('[SEP]')
      # Adjust the indexing based on your model's prediction
      result = parts[prediction + 1]  # +1 because index 0 is the question

      res = {'non_redundant': result, 'entity': parts[1], 'other_entity': parts[2]}

      results.append(res)

    return results

def classify_entities_and_return_parameters(input_text):
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

    res = {'non_redundant': result, 'entity': parts[1], 'other_entity': parts[2]}

    return res

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
            # if the term is found with a space before and after, it is a whole word match and not a substring
            if (start == 0 or text[start - 1] == ' ') and (end == len(text) or text[end] == ' '):
              results.append({"Text": term, "Type": taxonomy_list[0], "BeginOffset": start, "EndOffset": end})
              
            start += len(term)  # Move start index beyond the current word to avoid overlapping matches
    return results

climate_custom_taxonomy = ['custom_taxonomy', 'climate change', 'zero waste policy', 'zero waste', 'zero hunger', 'zero carbon buildings', 'wind power', 'wind energy industry', 'wind energy', 'wildfire management', 'wholesale energy markets', 'whistleblower protection', 'wetland conservation', 'weee', 'water utilities', 'water usage', 'water resource management', 'water quality', 'water framework directive', 'water conservation', 'waste reduction', 'waste management industry', 'waste management', 'waste electrical and electronic equipment directive', 'waste', 'vulnerability management', 'vulnerability', 'volunteering', 'volunteer programs', 'volunteer management', 'vms', 'vehicle emissions', 'vector control', 'vbp', 'various services and facilities', 'various medicaid waiver programs', 'variational autoencoders', 'value-based purchasing', 'value proposition', 'valuation cap', 'vaes', 'vaccination programs', 'utilities and grid operators', 'urban sustainability', 'urban heat island effect', 'urban green spaces', 'unmanned aerial vehicles', 'united nations sustainable development goals', 'unified endpoint management', 'unfccc', 'uem', 'uavs', 'triple bottom line', 'triage', 'transportation sector', 'transportation safety regulations', 'transportation policy', 'transportation planning', 'transportation infrastructure financing', 'transportation demand management', 'transportation', 'transport layer security', 'transparent reporting', 'transparent governance', 'transparency regulations', 'transparency and accountability', 'transmission and distribution', 'training and development', 'traffic safety', 'traffic management', 'traffic engineering', 'traffic congestion', 'trademark registration', 'trademark', 'trade compliance', 'traction', 'tourism and hospitality industry', 'tornado preparedness', 'tls', 'time series analysis', 'threat intelligence', 'theory of change', 'term sheet', 'telehealth', 'techstars', 'technological hazards', 'team building', 'tdm', 'tcfd', 'tax compliance', 'tax advisory', 'task force on climate-related financial disclosures', 'talent management', 'talent acquisition', 'synthetic data', 'sweat equity', 'sustainable water management', 'sustainable urban development', 'sustainable transport', 'sustainable tourism', 'sustainable textiles', 'sustainable technologies', 'sustainable supply chain', 'sustainable revenue', 'sustainable product design', 'sustainable packaging', 'sustainable mining', 'sustainable marine resources', 'sustainable livelihoods', 'sustainable land use', 'sustainable innovation', 'sustainable forestry', 'sustainable food systems', 'sustainable fisheries management', 'sustainable fisheries act', 'sustainable fisheries', 'sustainable finance disclosure regulation', 'sustainable finance', 'sustainable energy systems', 'sustainable energy', 'sustainable development indicators', 'sustainable development goals', 'sustainable consumption', 'sustainable compliance', 'sustainable communities', 'sustainable cities and communities', 'sustainable business practices', 'sustainable agriculture', 'sustainability training', 'sustainability reporting', 'sustainability in it operations', 'sustainability audits', 'sustainability advisory', 'sustainability', 'supply chain transparency', 'supply chain security', 'supply chain management', 'supplier collaboration', 'summative evaluation', 'subways and metro systems', 'substance abuse prevention', 'style transfer', 'stt', 'strategic sourcing', 'strategic planning', 'stock options', 'steel industry', 'statistical analysis', 'standards to assess healthcare performance', 'standards compliance', 'stakeholder engagement', 'staff development', 'ssl', 'sri', 'solar power', 'solar energy industry', 'solar energy', 'soil quality', 'society', 'socially responsible investing', 'social vulnerability', 'social media strategy', 'social innovation', 'social impact', 'social engineering defense', 'social determinants of health', 'social and labor standards', 'soc', 'soar', 'snf', 'smart transportation', 'smart grids', 'smart buildings', 'slas', 'skills training', 'skilled nursing facility', 'siem', 'shipping and maritime industry', 'shipping', 'shelter management', 'sharing of electronic health information', 'shares', 'shareholder agreement', 'shale gas', 'sfdr', 'sexual and reproductive health', 'security auditing', 'search and rescue', 'sea level rise', 'sdn', 'sdgs', 'SDG alignment', 'scope III', 'scope II', 'scope I', 'scope 3', 'scope 2', 'scope 1', 'scientometrics', 'scientist', 'SASB', 'sar', 'sanitation and hygiene', 'sales strategy', 'saas', 'run rate', 'rps', 'rpa', 'rohs', 'robotic process automation', 'road transportation', 'road signs and markings', 'road pricing', 'road maintenance', 'rmf', 'risk management framework', 'risk management', 'risk communication', 'risk assessment', 'revenue model', 'return on investment', 'retail industry', 'restructuring advisory', 'restriction of hazardous substances directive', 'responsible investment', 'responsible consumption and production', 'responsible advertising', 'resource management', 'resource efficiency initiative', 'resilience planning', 'resilience', 'research utilization', 'research uptake', 'research translation', 'research strategy', 'research quality assessment', 'research productivity', 'research policy', 'research performance indicators', 'research output', 'research metrics', 'research integrity', 'research institutions', 'research influence', 'research impact', 'research governance', 'research evaluation', 'research ethics', 'research dissemination', 'research and development', 'renewables', 'renewable portfolio standards', 'renewable heat', 'renewable energy sources', 'renewable energy integration', 'renewable energy industry', 'renewable energy directive', 'renewable energy certificates', 'renewable energy', 'remote work', 'remote healthcare services', 'reinforcement learning', 'regulatory technology', 'regulatory strategy', 'regulatory risk management', 'regulatory reporting', 'regulatory policy', 'regulatory liaison', 'regulatory intelligence', 'regulatory impact analysis', 'regulatory frameworks', 'regulatory compliance', 'regulatory change management', 'regulatory auditing', 'regulatory affairs', 'regulatory advisory', 'regtech', 'regression analysis', 'reduced inequalities', 'red teaming', 'recycling', 'recs', 'reconstruction', 'real estate development', 'readiness for health emergencies', 'ransomware protection', 'rail transportation', 'radiological emergency', 'question answering', 'quantum computing', 'quantitative evaluation', 'quality payment program', 'quality education', 'quality assurance', 'qualitative evaluation', 'qpp', 'purple teaming', 'publication analysis', 'public transportation', 'public relations advisory', 'public key infrastructure', 'public health surveillance', 'public health nutrition', 'public health nursing', 'public health law', 'public health genomics', 'public health ethics', 'public health emergency preparedness', 'public health emergency', 'public health administration', 'public health', 'public disclosure of environmental data policies', 'public cloud', 'psychological first aid', 'prototype', 'protected health information', 'proptech', 'proof of concept', 'prompt engineering', 'project management', 'program evaluation', 'profit margin', 'product responsibility', 'product launch', 'product development advisory', 'product development', 'procurement advisory', 'process evaluation', 'private cloud', 'privacy regulations', 'predictive analytics', 'pqrs', 'ppo', 'ppas', 'power purchase agreements', 'power plants', 'power generation', 'ports', 'population health', 'pollution prevention', 'pollution control', 'polluter pays principle policies', 'policy implementation', 'policy development', 'policies and regulations', 'platform as a service (paas)', 'platform as a service', 'pki', 'physician quality reporting system', 'phishing prevention', 'philanthropy', 'phi', 'pharmaceutical industry', 'pfa', 'permaculture', 'performance improvement', 'performance engineering', 'people, planet, profit', 'penetration testing', 'peer review', 'peaking power', 'peace, justice, and strong institutions', 'pdp', 'patient safety', 'patent filing', 'patent', 'passenger rail', 'passenger experience', 'partnerships for the goals', 'partnership', 'paris agreement', 'paas', 'outpatient surgical facilities', 'outcome evaluation', 'organizational development', 'organizational culture', 'organic farming', 'opioid epidemic', 'operations consulting', 'open data', 'oil production', 'oil and gas industry', 'oil and gas exploration', 'ocean conservation', 'ocean acidification', 'occupational health', 'nutrition and dietetics', 'nuclear power', 'nuclear energy', 'nrf', 'no poverty', 'nlp', 'nih', 'neural networks', 'networking events', 'nepa', 'negative emissions technologies', 'ndcs', 'ndc codes', 'nbsaps', 'nature-based solutions', 'natural resource management', 'natural gas', 'natural disasters', 'nationally determined contributions', 'national response framework', 'national institutes of health', 'national environmental policy act', 'national drug codes', 'national biodiversity strategies and action plans', 'national adaptation plans', 'naps', 'msc', 'mro', 'mpas', 'montreal protocol', 'mobile security', 'mobile application development', 'mixed-methods evaluation', 'mitigation strategies', 'mips', 'mining industry', 'microservices architecture', 'microservices', 'microgrids', 'mfa', 'merit-based incentive payment system', 'merger and acquisition advisory', 'mental health parity act', 'mental health', 'medicare-covered home health services', 'medicare program parts', 'medicare part a/b/c/d', 'medicare access and chip reauthorization act', 'medical surge', 'medicaid waivers', 'media and entertainment industry', 'measures to prevent harm in healthcare settings', 'measurement ', 'mdr', 'mci', 'maternal and child health', 'mass transit', 'mass casualty incident', 'mass care', 'maritime shipping', 'marine stewardship council certification', 'marine stewardship council', 'marine protected areas', 'manufacturing industry', 'management', 'managed detection and response', 'man-made disasters', 'macra', 'lrt', 'low-emission fuels', 'low-carbon economy', 'low carbon fuels', 'low carbon fuel standard', 'low carbon economy', 'long-term profitability', 'long-term care', 'logistics', 'logic models', 'local sourcing', 'lng', 'liquefied natural gas', 'limited partner', 'limited liability company (llc)', 'light rail transit', 'life on land', 'life cycle assessment', 'life below water', 'licensing and permits', 'lessons learned', 'legislation related to mental health coverage', 'legal compliance', 'legal advisory', 'legacy system modernization', 'lean startup', 'leadership development', 'leadership', 'lca', 'late-stage startup', 'last mile delivery', 'land use planning policies', 'land use planning', 'labor rights', 'labor and employment regulations', 'kyoto protocol', 'kyc', 'kubernetes', 'knowledge transfer', 'knowledge mobilization', 'knowledge management', 'know your customer', 'itsm', 'irena', 'ips', 'ipo', 'ipcc', 'ip', 'iot', 'investor dialogue', 'investor', 'investment advisory', 'intrusion prevention systems', 'intrusion detection systems', 'intrapreneurship', 'international renewable energy agency framework', 'international civil aviation organization', 'internal controls', 'internal communications', 'intermodal transportation', 'intergovernmental panel on climate change', 'intelligent transportation systems', 'intellectual property rights', 'intellectual property regulations', 'intellectual property', 'insurance industry', 'insider threat management', 'innovation management', 'innovation culture', 'innovation', 'inland waterways', 'injury prevention', 'infrastructure resilience', 'infrastructure as a service (iaas)', 'infrastructure as a service', 'information security', 'infectious disease control', 'industry, innovation, and infrastructure', 'industry standards', 'industry conference', 'industries and applications', 'industrial ecology', 'industrial control systems', 'indian health service', 'incubator', 'incorporation', 'incident response', 'incident command system', 'improvement opportunities', 'import regulations', 'impact investing', 'impact factor', 'impact', 'image super-resolution', 'image generation', 'illegal practices in the healthcare system', 'ihs', 'ifc taxonomies', 'ids', 'identity and access management', 'ics', 'icao', 'iam', 'iaas', 'hydropower industry', 'hydropower', 'hydrogen economy', 'hurricane response', 'human rights in supply chains', 'human rights', 'human resources consulting', 'home health services', 'hmo', 'hitech act', 'hipaa', 'highways', 'high-speed rail', 'high availability', 'hhs', 'helicopters', 'heavy industry', 'heat waves', 'healthtech', 'healthcare quality measures', 'healthcare fraud', 'healthcare financing', 'healthcare common procedure coding system', 'health systems', 'health services research', 'health promotion', 'health policy', 'health outcomes of a group of individuals', 'health maintenance organization', 'health literacy', 'health insurance portability and accountability act', 'health insurance continuation', 'health information technology for economic and clinical health act', 'health information exchange', 'health informatics', 'health impact assessment', 'health equity', 'health education', 'health economics', 'health disparities', 'health communication', 'health care sector', 'health and safety regulations', 'health and safety', 'health advocacy', 'hcpcs codes', 'hazards', 'hazardous waste', 'habitat restoration', 'habitat preservation', 'h-index', 'grid modernization', 'grid infrastructure', 'GRI, SASB, TCFD', 'gri', 'greentech', 'greenhouse gases', 'greenhouse gas emissions', 'greenhouse gas accounting', 'green technologies', 'green procurement policies', 'green procurement', 'green new deal', 'green marketing', 'green logistics', 'green jobs', 'green it', 'green infrastructure', 'green hydrogen', 'green finance', 'green energy transition', 'green economy', 'green chemistry', 'green building standards', 'green building', 'green bonds', 'grc', 'grant reviews', 'government and public sector', 'governance, risk, and compliance', 'go-to-market strategy', 'global warming', 'global reporting initiative standards', 'global health security', 'global health', 'glacier retreat', 'gis', 'geothermal energy industry', 'geothermal energy', 'geospatial data', 'general partner', 'general aviation', 'gender equality', 'fuel efficiency', 'fsc', 'freight rail', 'fqhc', 'fp&a', 'founder', 'fossil fuels', 'formative evaluation', 'forest stewardship council certification', 'forest stewardship council', 'forest management', 'food safety', 'food and drug administration', 'food and beverage industry', 'floods', 'floodplain management', 'flight safety', 'flight planning', 'flight operations', 'fixed-wing aircraft', 'fits', 'financial services and green finance', 'financial regulations', 'financial planning and analysis', 'financial model', 'financial assistance programs', 'financial advisory', 'ferry services', 'feed-in tariffs', 'federally qualified health center', 'federal it modernization', 'federal health', 'federal aviation administration', 'feature engineering', 'fda', 'fair trade', 'fair labor practices', 'faa', 'extreme weather events', 'extract, transform, load', 'export control regulations', 'explainable ai', 'exit strategy', 'evs', 'evidence-based policy', 'evaluation utilization', 'evaluation standards', 'evaluation reporting', 'evaluation frameworks', 'evaluation ethics', 'evaluation capacity building', 'evacuation', 'european union aviation safety agency', 'european green deal', 'eu taxonomy regulation', 'ets', 'etl', 'ethical standards', 'ethical sourcing', 'ethical marketing', 'ethical leadership', 'ethical investing', 'ethical financial management', 'ethical compliance', 'ethical business practices', 'ethical ai', 'esg advisory', 'esg reporting', 'esg criteria', 'esg', 'equal access to healthcare services', 'epidemiology', 'eoc', 'environmental, social, and governance', 'environmental stewardship', 'environmental services', 'environmental risk assessment', 'environmental remediation', 'environmental regulations', 'environmental protection act', 'environmental policy', 'environmental performance', 'environmental monitoring', 'environmental management systems', 'environmental laws', 'environmental law', 'environmental justice', 'environmental impact assessment regulations', 'environmental impact assessment', 'environmental impact', 'environmental health', 'environmental ethics', 'environmental education', 'environmental compliance', 'environmental certification', 'environment ', 'environment architecture', 'engineer', 'energy transition', 'energy trading', 'energy supply chain', 'energy supply', 'energy subsidies', 'energy storage industry', 'energy storage', 'energy security', 'energy risk management', 'energy resilience', 'energy regulatory framework', 'energy regulations', 'energy policy act', 'energy policy', 'energy mix', 'energy markets', 'energy management systems', 'energy justice', 'energy innovation', 'energy infrastructure', 'energy independence', 'energy import', 'energy governance', 'energy financing', 'energy export', 'energy efficiency directive', 'energy efficiency', 'energy education and awareness', 'energy economics', 'energy diplomacy', 'energy demand forecasting', 'energy consumption patterns', 'energy conservation building code', 'energy conservation', 'energy audits', 'energy auctions', 'energy access', 'energy', 'endangered species act', 'endangered species', 'encryption', 'ems', 'employment agreement', 'employee well-being', 'employee turnover', 'employee involvement', 'employee hiring', 'employee health and safety', 'emotion recognition', 'emissions trading system', 'emissions trading', 'emissions control', 'emissions', 'emission reduction targets', 'emergency response', 'emergency preparedness', 'emergency operations center', 'emergency alert system', 'emas', 'electronic health records', 'electricity generation', 'electricity', 'electric vehicles', 'electric vehicle (ev) industry', 'electric utilities', 'elasticity', 'eia', 'ehr', 'education and research institutions', 'edtech', 'edr', 'edge computing', 'ecotourism', 'ecosystem-based adaptation', 'ecosystem services', 'economic and social factors influencing health', 'economic advisory', 'ecological restoration', 'ecological footprint', 'eco-management and audit scheme', 'eco-labelling schemes', 'eco-labeling', 'eco-innovation', 'ecbc', 'easa', 'eas', 'earthquake preparedness', 'early-stage startup', 'early warning systems', 'e.g., nist, iso/iec 27001', 'e.g., leed', 'due diligence', 'drug pricing program', 'drought contingency', 'drought', 'drone operations', 'drgs', 'draas', 'donations management', 'dlp', 'diversity, equity, and inclusion (DEI)', 'diversity metrics', 'diversity and inclusion', 'distributed energy resources (der)', 'distributed energy resources', 'disruptive technology', 'disease prevention', 'disaster recovery planning', 'disaster recovery as a service', 'disaster recovery', 'disaster mitigation', 'disaster management', 'disaster declaration', 'disaster communications', 'digital transformation', 'digital marketing', 'differences in health outcomes among populations', 'diagnostic analytics', 'diagnosis-related groups', 'devops', 'descriptive analytics', 'ders', 'der', 'department of health and human services', 'demo day', 'demand response', 'dei', 'deforestation', 'decent work and economic growth', 'decarbonization strategies', 'decarbonization pathways', 'decarbonization', 'data analytics', 'damage assessment', 'd.e.i', 'current procedural terminology', 'csrd', 'csr', 'cspm', 'crowdfunding', 'critical infrastructure', 'crisis related to opioid misuse', 'crisis mapping', 'crisis management', 'cpt codes', 'cost-effectiveness analysis', 'cost-benefit analysis', 'cost savings', 'corsia', 'corporation', 'corporate sustainability reporting directive', 'corporate sustainability', 'corporate social responsibility', 'corporate philanthropy', 'corporate governance', 'corporate ethics', 'corporate citizenship', 'convertible note', 'convention on biological diversity', 'contract negotiation', 'continuous monitoring', 'content generation', 'content delivery network', 'containerization', 'consumer protection regulations', 'consumer goods industry', 'consolidated omnibus budget reconciliation act', 'conservation agriculture', 'conservation', 'connected vehicles', 'conflict minerals', 'compliance training', 'compliance monitoring', 'compliance management', 'compliance culture', 'compliance analytics', 'compliance advisory', 'competitive analysis', 'competition law', 'commuter rail', 'community resilience', 'community outreach', 'community impact', 'community health', 'community engagement', 'community development', 'commercial aviation', 'cobra', 'coal power', 'coal industry', 'coal and coal technologies', 'co-working space', 'co-authorship analysis', 'clinical modification', 'climate vulnerability', 'climate technology', 'climate smart agriculture', 'climate risk assessment', 'climate resilience', 'climate refugees', 'climate policy', 'climate neutrality', 'climate models', 'climate modeling', 'climate justice', 'climate impact assessment', 'climate finance', 'climate education', 'climate data', 'climate communication', 'climate change scenarios', 'climate change monitoring', 'climate change mitigation', 'climate change indicators', 'climate change governance', 'climate change ethics', 'climate change education', 'climate change communication', 'climate change and water resources', 'climate change and health', 'climate change and biodiversity', 'climate change adaptation', 'climate change act', 'climate adaptation strategies', 'climate action plans', 'climate action', 'climate', 'clean water and sanitation', 'clean water act', 'clean technology', 'clean power plan', 'clean energy technologies', 'clean energy standard', 'clean air act', 'classification', 'citizen-centric services', 'citation analysis', 'circular economy action plan', 'circular economy', 'circular business models', 'chronic disease management', 'chip', 'chemical spills', 'chemical industry', 'change management', 'centers for medicare & medicaid services', 'centers for disease control and prevention', 'cement industry', 'cdn', 'cdc', 'ccs', 'cbd', 'cargo aircraft', 'carbon trading', 'carbon tax', 'carbon sequestration', 'carbon reduction projects', 'carbon pricing mechanisms', 'carbon pricing', 'carbon offsetting and reduction scheme for international aviation', 'carbon offsetting', 'carbon offsets', 'carbon offset', 'carbon neutrality', 'carbon markets', 'carbon intensity', 'carbon footprint', 'carbon disclosure', 'carbon credits', 'carbon credit', 'carbon capture and storage', 'capital raise', 'capacity markets', 'california global warming solutions act', 'cabin crew', 'business transformation', 'bundled payments for care improvement', 'building and construction industry', 'brt', 'bridges', 'break-even point', 'brand strategy', 'bpci', 'board diversity', 'blockchain technology', 'biostatistics', 'biomass energy industry', 'biomass energy', 'biological threats', 'biofuel industry', 'bioenergy', 'biodiversity loss', 'biodiversity conservation', 'biodiversity', 'big data analytics', 'bibliometrics', 'bias mitigation', 'benchmarking', 'behavioral health', 'behavioral analytics', 'battery technologies', 'battery manufacturing', 'basel convention on hazardous wastes', 'base load power', 'avs', 'aviation weather', 'aviation technology', 'aviation sustainability', 'aviation security', 'aviation safety', 'aviation regulations', 'aviation noise', 'aviation maintenance repair and overhaul', 'aviation law', 'aviation insurance', 'aviation industry', 'aviation fuel', 'aviation finance', 'aviation emissions', 'aviation communication', 'aviation accidents', 'aviation', 'autonomous vehicles', 'automotive industry', 'audit and assurance services', 'atm', 'atc', 'artificial intelligence', 'arctic amplification', 'application security', 'api management', 'antivirus and antimalware', 'anti-money laundering', 'anti-corruption policies', 'anti-corruption', 'anti-bribery and corruption', 'anomaly detection', 'aml', 'ambulatory surgical centers', 'altmetrics', 'alternative fuels', 'airport', 'airports', 'airport security', 'airport operations', 'airport infrastructure', 'airlines', 'airline routes', 'airline revenue management', 'airline operations', 'airline fleet', 'airline codeshare', 'airline alliances', 'aircraft manufacturing', 'aircraft maintenance', 'aircraft leasing', 'aircraft design', 'aircraft', 'air traffic management', 'air traffic control', 'air quality', 'air pollution control', 'air cargo', 'ai for accessibility', 'ai ethics and governance', 'ai', 'agroecology', 'agriculture sector', 'aging and geriatrics', 'afforestation', 'affordable and clean energy', 'advisory', 'advisor', 'advanced persistent threats', 'adolescent health', 'adaptation financing', 'aco', 'accountable care organization', 'accelerator', 'academic collaboration', 'abc', 'ab 32', 'a/b testing', 'a.i.', 'drones', 'drone', 'wind', 'solar', 'plastic', 'plastics', 'supply chain', 'net-zero']
sustainability_custom_taxonomy = ['sustainability', 'hmo', 'logistics', 'legislation related to mental health coverage', 'startup culture', 'traffic management', 'public relations advisory', 'adaptation financing', 'ecbc', 'fuel efficiency', 'cloud backup', 'road transportation', 'intelligent transportation systems', 'regtech', 'energy security', 'research influence', 'ai in design', 'airline revenue management', 'mental health', 'geographic information systems) in disaster management', 'data security', 'import regulations', 'geospatial data', 'health advocacy', 'creativity in ai', 'light rail transit', 'health literacy', 'process evaluation', 'differences in health outcomes among populations', 'circular economy', 'ocean acidification', 'climate change scenarios', 'sea level rise', 'evaluation utilization', 'data cleansing', 'polluter pays principle policies', 'energy', 'ssl', 'transport layer security', 'climate change monitoring', 'electric vehicles', 'fintech', 'blue teaming', 'gis', 'data governance', 'electric vehicle (ev) industry', 'zero-shot learning', 'renewable energy industry', 'climate education', 'sustainable fisheries', 'health promotion', 'esg', 'energy financing', 'aviation maintenance repair and overhaul', 'climate', 'telehealth', 'endpoint security', 'environmental services', 'remote work', 'intergovernmental panel on climate change', 'circular business models', 'department of health and human services', 'global health security', 'green economy', 'evaluation standards', 'utilities and grid operators', 'blockchain technology', 'health and safety regulations', 'energy policy', 'wildfire management', 'valuation cap', 'series a/b/c funding', 'talent management', 'tornado preparedness', 'operations consulting', 'transfer learning', 'procurement advisory', 'early-stage startup', 'responsible consumption and production', 'technological hazards', 'financial regulations', 'highways', 'vendor management', 'health informatics', 'renewable energy sources', 'flight safety', 'bibliometrics', 'devops', 'aviation noise', 'internal controls', 'unified endpoint management', 'resilience', 'proof of concept', 'environmental ethics', 'infrastructure resilience', 'green jobs', 'public transportation', 'reinforcement learning', 'bias mitigation', 'employee well-being', 'aviation safety', 'sustainable marine resources', 'public health emergency preparedness', 'innovation', 'climate change and health', 'ferry services', 'evacuation', 'ccs', 'research performance indicators', 'natural language processing', 'disease prevention', 'eoc', 'green energy transition', 'edge computing', 'stock options', 'environmental impact assessment', 'evs', 'user-centered design', 'atm', 'sustainable livelihoods', 'labor and employment regulations', 'sentiment analysis', 'vehicle emissions', 'climate change and water resources', 'public health administration', 'strategic sourcing', 'power generation', 'drought', 'unique selling proposition (usp)', 'sustainable supply chain', 'cloud devops', 'social vulnerability', 'ai', 'mpas', 'biodiversity conservation', 'esg) advisory', 'environmental performance', 'coal power', 'minimum viable product', 'man-made disasters', 'financial assistance programs', 'customer segmentation', 'measures to prevent harm in healthcare settings', 'healthcare fraud', 'ipcc', 'regulatory reporting', 'recycling', 'sustainable food systems', 'series d funding', 'health systems', 'traffic engineering', 'vector control', 'energy conservation building code', 'vpc', 'biological threats', 'tourism and hospitality industry', 'green hydrogen', 'compliance advisory', 'synthetic data', 'impact factor', 'eu taxonomy regulation', 'liquefied natural gas', 'transportation', 'cloud collaboration tools', 'climate change education', 'corporate citizenship', 'information security', 'electricity', 'energy diplomacy', 'eco-management and audit scheme', 'legacy system modernization', 'traffic safety', 'ndc codes', 'data integration', 'community outreach', 'cybersecurity regulations', 'pki', 'gender equality', 'air traffic management', 'time series analysis', 'language models', 'saas', 'multi-cloud', 'dlp', 'cloud compliance', 'greenhouse gases', 'impact investing', 'user experience (ux)', 'business process reengineering', 'regulatory policy', 'financial planning and analysis', 'ifc taxonomies', 'tunnels', 'transparency regulations', 'data recovery', 'equity financing', 'data anonymization', 'it service management', 'renewable heat', 'maternal and child health', 'regulatory technology', 'mining industry', 'aircraft maintenance', 'data pipeline', 'cabin crew', 'rps', 'green chemistry', 'building and construction industry', 'grc', 'sustainable water management', 'national adaptation plans', 'industrial control systems', 'market entry strategy', 'injury prevention', 'esg reporting', 'staff development', 'carbon pricing mechanisms', 'local sourcing', 'renewable energy certificates', 'legal compliance', 'environmental health', 'emergency alert system', 'last mile delivery', 'climate neutrality', 'hitech act', 'lrt', 'search and rescue', 'sri', 'business intelligence', 'social impact', 'energy audits', 'industrial ecology', 'climate policy', 'climate models', 'waste electrical and electronic equipment directive', 'airline codeshare', 'data catalog', 'knowledge mobilization', 'current procedural terminology', 'water utilities', 'text summarization', 'ips', 'tcfd', 'standards compliance', 'cloud monitoring', 'uavs', 'intellectual property rights', 'medicare part a/b/c/d', 'cloud cost management', 'federal health', 'emas', 'veterans affairs health services', 'public health emergency', 'shareholder agreement', 'rmf', 'restructuring advisory', 'energy auctions', 'ai for accessibility', 'corsia', 'cloud data encryption', 'vulnerability management', 'nbsaps', 'startup office', 'research policy', 'enterprise architecture', 'retail industry', 'carbon intensity', 'intermodal transportation', 'healthcare quality measures', 'mdr', 'private cloud', 'geothermal energy', 'disaster communications', 'sustainable transport', 'behavioral health', 'regulatory impact analysis', 'standards to assess healthcare performance', 'mixed-methods evaluation', 'strategic planning', 'health services research', 'peaking power', 'sustainability reporting', 'aviation insurance', 'data modeling', 'sustainable agriculture', 'human rights in supply chains', 'leadership', 'evaluation ethics', 'cloud security posture management', 'marine stewardship council', 'sexual and reproductive health', 'climate change', 'carbon offsetting and reduction scheme for international aviation', 'intellectual property', 'ip', 'scope 1', 'scope I', 'scope II', 'scope 2', 'scope 3', 'scope III', 'renewable portfolio standards', 'research quality assessment', 'carbon offsetting', 'carbon offset', 'carbon offsets', 'carbon credit', 'carbon credits', 'hurricane response', 'startup mentor', 'image generation', 'cspm', 'seed round', 'airports', 'cloud management', 'data archiving', 'cloud integration', 'aviation accidents', 'mfa', 'management consulting', 'subways and metro systems', 'anomaly detection', 'machine learning', 'gri', 'climate finance', 'hcpcs codes', 'urban sustainability', 'business transformation', 'series b funding', 'deforestation', 'cloud orchestration', 'centers for medicare & medicaid services', 'feed-in tariffs', 'startup accelerator', 'climate change and biodiversity', 'ports', 'flight operations', 'aviation industry', 'energy risk management', 'board member', 'threat intelligence', 'research ethics', 'cloud app development', 'startup valuation', 'waste management', 'triage', 'freight rail', 'purple teaming', 'carbon pricing', 'high availability', 'mitigation strategies', 'distributed energy resources (der)', 'united nations sustainable development goals', 'cybersecurity advisory', 'infrastructure as a service (iaas)', 'smart grids', 'green infrastructure', 'sfdr', 'extract, transform, load', 'environmental justice', 'low carbon economy', 'carbon neutrality', 'platform as a service', 'application security', 'forest management', 'data mining', 'research governance', 'multimodal models', 'lean startup', 'customer satisfaction', 'performance engineering', 'hydrogen economy', 'csrd', 'adolescent health', 'energy transition', 'clean water act', 'research evaluation', 'medicaid waivers', 'community health', 'chemical spills', 'vaccination programs', 'grid infrastructure', 'eco-labeling', 'green procurement', 'diagnosis-related groups', 'air traffic control', 'habitat restoration', 'waste management industry', 'mvp (minimum viable product)', 'mass care', '5g network integration', 'community resilience', 'tax advisory', 'kubernetes', 'airline alliances', 'consumer protection regulations', 'training and development', 'product responsibility', 'wind energy industry', 'engineer', 'value proposition', 'unfccc', 'edr', 'california global warming solutions act', 'aviation communication', 'capacity markets', 'pitch competition', 'data stewardship', 'soar', 'earthquake preparedness', 'land use planning policies', 'ambulatory surgical centers', 'equity stake', 'cloud performance optimization', 'regulatory affairs', 'microservices architecture', 'theory of change', 'wind power', 'y combinator', 'ppo', 'health information technology for economic and clinical health act', 'trademark', 'radiological emergency', 'disaster recovery', 'ndcs', 'government and public sector', 'siem', 'security information and event management', 'sustainable product design', 'formative evaluation', 'climate impact assessment', 'business plan', 'emergency response', 'eia', 'compliance training', 'cloud databases', 'carbon tax', 'phi', 'water framework directive', 'data standardization', 'image super-resolution', 'public health', 'aviation finance', 'export control regulations', 'innovation management', 'climate communication', 'unmanned aerial vehicles', 'pricing strategy', 'environmental management systems', 'stakeholder engagement', 'air cargo', 'sustainable finance', 'grid modernization', 'seed capital', 'energy trading', 'disaster declaration', 'cloud migration', 'ethical investing', 'crisis mapping', 'agriculture sector', 'few-shot learning', 'prescriptive analytics', 'venture round', 'clean energy standard', 'transparency and accountability', 'energy resilience', 'climate smart agriculture', 'ipo', 'tdm', 'national response framework', 'eco-innovation', 'disaster recovery as a service', 'ets', 'fqhc', 'ecosystem-based adaptation', 'volunteer programs', 'startup advisor', 'disaster recovery planning', 'ecological restoration', 'national biodiversity strategies and action plans', 'carbon capture and storage', 'labor rights', 'data center consolidation', 'financial advisory', 'user acquisition', 'merger and acquisition advisory', 'iam', 'paris agreement', 'sustainable tourism', 'clean air act', 'energy management systems', 'fp&a', 'revenue-based financing', 'sustainable compliance', 'renewable energy', 'sdn', 'nutrition and dietetics', 'climate action plans', 'oil and gas exploration', 'data sampling', 'project management', 'floods', 'software defined networking', 'nepa', 'aviation fuel', 'advisory', 'natural resource management', 'public cloud', 'cdn', 'epidemiology', 'waste reduction', 'energy economics', 'life cycle assessment', 'healthcare financing', 'ppas', 'regulatory change management', 'software as a service (saas)', 'scientist', 'drug pricing program', 'crisis management', 'faa', 'food and drug administration', 'data masking', 'renewable energy integration', 'cyber threat hunting', 'advanced persistent threats', 'break-even point', 'greenhouse gas accounting', 'h-index', 'under medicare part d', 'ai in gaming', 'mass transit', 'internet of things', 'pre-seed funding', 'readiness for health emergencies', 'sustainable energy systems', 'industry conference', 'quality assurance', 'nuclear power', 'aviation', 'explainable ai', 'energy education and awareness', 'autonomous vehicles', 'market research', 'marketing strategy', 'electronic health records', 'ics) security', 'performance review', 'soc', 'evidence-based policy', 'data mart', 'fault tolerance', 'real estate development', 'compliance monitoring', 'profit margin', 'green bonds', 'energy demand forecasting', 'conservation', 'ai ethics and governance', 'public health genomics', 'traffic congestion', 'public key infrastructure', 'security policy management', 'it consulting', 'mci', 'home health services', 'rpa', 'emission reduction targets', 'energy governance', 'environmental stewardship', 'environmental compliance', 'content delivery network', 'venture debt', 'drone operations', 'cyber resilience', 'bioenergy', 'nlp', 'battery manufacturing', 'long-term care', 'accelerator', 'endangered species act', 'secure socket layer', 'managed detection and response', 'qualitative evaluation', 'low carbon fuels', 'green marketing', 'research uptake', 'general partner', 'reconstruction', 'incorporation', 'limited partner', 'market fit', 'energy independence', 'customer acquisition', 'fda', 'cloud load balancing', 'ihs', 'financial services and green finance', 'hydropower industry', 'self-supervised learning', 'passenger experience', 'weee', 'fixed-wing aircraft', 'pivot', 'environmental, social, and governance', 'carbon offsets', 'privacy regulations', 'energy conservation', 'water conservation', 'corporation', 'sustainable fisheries act', 'transportation planning', 'health education', 'bpci', 'api management', 'cloud storage', 'pdp', 'funding mechanisms for healthcare services', 'deep learning', 'health impact assessment', 'aircraft design', 'employee hiring', 'battery technologies', 'substance abuse prevention', 'virtual machines', 'zero trust architecture', 'forest stewardship council', 'environmental protection act', 'performance improvement', 'environmental risk assessment', 'transportation sector', 'education and research institutions', 'iot) security', 'edtech', 'data warehousing', 'mobile security', 'health care sector', 'regulatory liaison', 'aviation emissions', 'avs', 'ecosystem services', 'social determinants of health', 'contract negotiation', 'growth hacking', 'industries and applications', 'diversity and inclusion', 'natural gas', 'sustainable communities', 'international civil aviation organization', 'health and safety', 'sustainable development indicators', 'semi-supervised learning', 'convertible note', 'cost-benefit analysis', 'market analysis', 'business accelerator', 'forest stewardship council certification', 'quantitative evaluation', 'donations management', 'regulatory advisory', 'aml', 'heat waves', 'public health ethics', 'it budgeting and cost management', 'environmental impact', 'airline operations', 'biomass energy industry', 'generative ai', 'outcome evaluation', 'cargo aircraft', 'pollution control', 'smart buildings', 'xai', 'distributed energy resources', 'passenger rail', 'manufacturing industry', 'business', 'networking events', 'consolidated omnibus budget reconciliation act', 'sustainable forestry', 'logic models', 'population health', 'public health surveillance', 'growth capital', 'water quality', 'legal advisory', 'organizational culture', 'power purchase agreements', 'economic and social factors influencing health', 'know your customer', 'capital raise', 'solar energy industry', 'green it', 'supply chain management', 'aviation regulations', 'agroecology', 'zero carbon buildings', 'smart transportation', 'occupational health', 'climate vulnerability', 'data lake', 'clinical modification', 'business development', 'serial entrepreneur', 'green procurement policies', 'crowdfunding', 'biodiversity', 'citation analysis', 'private equity', 'partnership', 'aircraft', 'sanitation and hygiene', 'pitch deck', 'corporate governance', 'environmental policy', 'sustainable mining', 'market disruption', 'text-to-speech', 'energy consumption patterns', 'vbp', 'automotive industry', 'team building', 'fine-tuning', 'management', 'policies and regulations', 'data analytics advisory', 'ecological footprint', 'startup law', 'mips', 'transmission and distribution', 'indian health service', 'healthcare common procedure coding system', 'lessons learned', 'ai in education', 'regulatory strategy', 'big data analytics', 'cyber security', 'sar', 'wind energy', 'climate change communication', 'uem', 'cloud native applications', 'slas', 'shipping', 'responsible investment', 'emergency operations center', 'co-working space', 'knowledge management', 'vulnerability', 'environmental certification', 'airline routes', 'penetration testing', 'emissions', 'emissions trading', 'energy infrastructure', 'climate data', 'chemical industry', 'anti-money laundering', 'prototype', 'montreal protocol', 'greenhouse gas emissions', 'cbd', 'security governance', 'health disparities', 'marine stewardship council certification', 'series e funding', 'sharing of electronic health information', 'kyc', 'various medicaid waiver programs', 'tech stack', 'electric utilities', 'data protection regulations', 'advisor', 'antivirus and antimalware', 'basel convention on hazardous wastes', 'health policy', 'generative adversarial networks', 'road signs and markings', 'angel funding', 'aviation security', 'limited liability company (llc)', 'green finance', 'customer retention', 'cyber forensics', 'climate change act', 'draas', 'natural disasters', 'global reporting initiative standards', 'alternative fuels', 'centers for disease control and prevention', 'go-to-market strategy', 'feature engineering', 'abc', 'cybersecurity frameworks', 'tts', 'carbon markets', 'clustering', 'regulatory compliance', 'green building', 'disaster mitigation', 'aviation technology', 'public health law', 'marketing consulting', 'citizen-centric services', 'co-authorship analysis', 'permaculture', 'community development', 'preferred provider organization', 'quantum computing', 'environment', 'apt) defense', 'scientometrics', 'der', 'triple bottom line', 'ems', 'risk management framework', 'research output', 'disruptive technology', 'rohs', 'aviation weather', 'sustainable packaging', 'coal and coal technologies', 'health communication', 'wealth management', 'cloud security', 'hydropower', 'virtualization', 'pollution prevention', 'venture partner', 'energy supply', 'corporate sustainability reporting directive', 'security operations center', 'shares', 'regulatory frameworks', 'telecommunications industry', 'emotion recognition', 'rotary-wing aircraft', 'drought contingency', 'convention on biological diversity', 'angel investor', 'business model', 'helicopters', 'ethical compliance', 'code generation', 'chronic disease management', 'insurance industry', 'energy storage', 'road maintenance', 'energy efficiency', 'itsm', 'music generation', 'organic farming', 'aco', 'opioid epidemic', 'data normalization', 'it skills development', 'corporate sustainability', 'energy storage industry', 'afforestation', 'icao', 'environmental education', 'incident command system', 'carbon disclosure', 'text translation', "children's health insurance program", 'family and friends round', 'data quality', 'resource efficiency initiative', 'vms', 'late-stage startup', 'climate modeling', 'robotic process automation', 'lng', 'continuous monitoring', 'energy access', 'bootstrapping', 'data visualization', 'encryption', 'environmental remediation', 'health equity', 'security auditing', 'water resource management', 'funding round', 'connected vehicles', 'cap table', 'incident response', 'data augmentation', 'enterprise cloud solutions', 'sustainability advisory', 'biofuel industry', 'competitive analysis', 'sustainable urban development', 'arctic amplification', 'artificial intelligence', 'data enrichment', 'patent', 'climate change mitigation', 'series a funding', 'easa', 'wetland conservation', 'sales strategy', 'climate refugees', 'due diligence', 'social media strategy', 'environmental monitoring', 'climate risk assessment', 'renewables', 'ecotourism', 'solar power', 'merit-based incentive payment system', 'transportation safety regulations', 'renewable energy directive', 'aviation sustainability', 'predictive analytics', 'road pricing', 'marine protected areas', 'green technologies', 'green new deal', 'data segmentation', 'green building standards', 'angel round', 'shelter management', 'data privacy', 'ders', 'phishing prevention', 'cloud governance', 'cloud networking', 'minimal viable product (mvp)', 'ethical business practices', 'evaluation frameworks', 'hazardous waste', 'healthtech', 'audit and assurance services', 'a/b testing', 'responsible advertising', 'mvp', 'open data', 'ethical marketing', 'eas', 'ai in marketing', 'floodplain management', 'medicare program parts', 'climate adaptation strategies', 'corporate social responsibility', 'general aviation', 'energy import', 'ethical standards', 'maritime shipping', 'containerization', 'incubator', 'research metrics', 'intellectual property regulations', 'data transformation', 'international renewable energy agency framework', 'aging and geriatrics', 'series c funding', 'vesting schedule', 'shale gas', 'pilot training', 'physician quality reporting system', 'quality payment program', 'circular economy action plan', 'conflict minerals', 'low-carbon economy', 'various services and facilities', 'speech-to-text', 'climate resilience', 'green logistics', 'scalability and elasticity', 'airport security', 'research dissemination', 'software as a service', 'debt financing', 'emissions trading system', 'urban heat island effect', 'cloud computing', 'startup ecosystem', 'global warming', 'european green deal', 'hhs', 'pharmaceutical industry', 'infrastructure as a service', 'digital transformation', 'cloud automation', 'psychological first aid', 'endpoint detection and response', 'peer review', 'eco-labelling schemes', 'iot', 'pqrs', 'red teaming', 'sustainable textiles', 'united nations framework convention on climate change', 'hipaa', 'mass casualty incident', 'medical surge', 'investment advisory', 'product development advisory', 'carbon sequestration', 'competition law', 'seed funding', 'drgs', 'early warning systems', 'health maintenance organization', 'regulatory intelligence', 'naps', 'multi-factor authentication', 'program evaluation', 'energy innovation', 'cost-effectiveness analysis', 'descriptive analytics', 'understanding of health information', 'bus rapid transit', 'fair trade', 'media and entertainment industry', 'va health system', 'startup pitch', 'atc', 'platform as a service (paas)', 'federally qualified health center', 'critical infrastructure', 'electricity generation', 'hazards', 'nature-based solutions', 'brand strategy', 'low carbon fuel standard', 'risk communication', 'neural networks', 'urban green spaces', 'conservation agriculture', 'style transfer', 'unsupervised learning', 'security orchestration, automation, and response', 'research productivity', 'public health nutrition', 'term sheet', 'regression analysis', 'burn rate', 'ransomware protection', 'vaes', 'tax compliance', 'licensing and permits', 'energy mix', 'chip', 'mobile application development', 'research integrity', 'health outcomes of a group of individuals', 'biomass energy', 'intrusion detection systems', 'public health nursing', 'prompt engineering', 'sustainable consumption', 'research translation', 'climate change adaptation', 'volunteer management', 'managed cloud services', 'proptech', 'skills training', 'statistical analysis', 'greentech', 'entrepreneurship', 'text generation', 'serverless computing', 'medicare-covered home health services', 'corporate ethics', 'airline fleet', 'disaster management', 'compliance culture', 'altmetrics', 'fossil fuels', 'social innovation', 'nih', 'commercial aviation', 'data retention', 'decarbonization pathways', 'energy supply chain', 'ai in writing', 'scaling', 'cyber hygiene', 'environmental regulations', 'supply chain security', 'revenue model', 'traction', 'whistleblower protection', 'compliance analytics', 'health insurance continuation', 'climate technology', 'carbon trading', 'evaluation reporting', 'ai in art', 'techstars', 'sustainable finance disclosure regulation', 'digital marketing', 'qpp', 'consumer goods industry', 'low-emission fuels', 'sustainable energy', 'data centers and it industry', 'public disclosure of environmental data policies', 'evaluation capacity building', 'ai in healthcare', 'airport infrastructure', 'restriction of hazardous substances directive', 'cyber risk management', 'conversational ai', 'sustainable business practices', 'research utilization', 'skilled nursing facility', 'virtual private cloud', 'microservices', 'mental health parity act', 'emergency preparedness', 'kyoto protocol', 'environmental impact assessment regulations', 'energy markets', 'illegal practices in the healthcare system', 'data loss prevention', 'airport operations', 'mezzanine financing', 'regulatory auditing', 'cx) consulting', 'data auditing', 'venture funding', 'oil and gas industry', 'diagnostic analytics', 'transportation policy', 'variational autoencoders', 'people, planet, profit', 'it infrastructure optimization', 'lca', 'federal aviation administration', 'ab 32', 'equity crowdfunding', 'human-ai interaction', 'food safety', 'recs', 'research strategy', 'energy regulatory framework', 'cloud scalability', 'macra', 'sustainable development goals', 'user interface (ui)', 'hybrid cloud', 'biostatistics', 'ehr', 'energy subsidies', 'sweat equity', 'medicare access and chip reauthorization act', 'transformer models', 'financial model', 'industry standards', 'sustainable fisheries management', 'knowledge transfer', 'question answering', 'oil production', 'iaas', 'commuter rail', 'csr', 'flight planning', 'irena', 'national environmental policy act', 'fundraising', 'human resources consulting', 'biodiversity loss', 'brt', 'equity', 'sustainability in it operations', 'geothermal energy industry', 'aircraft leasing', 'summative evaluation', 'publication analysis', 'economic advisory', 'coal industry', 'zero waste policy', 'federal it modernization', 'identity and access management', 'negative emissions technologies', 'cloud innovation', 'data encryption', 'ids', 'open source software adoption', 'decarbonization strategies', 'outpatient surgical facilities', 'prescription drug plan', 'non-disclosure agreement (nda)', 'regulatory risk management', 'etl', 'venture capitalist', 'rail transportation', 'intrusion prevention systems', 'energy regulations', 'aviation law', 'zero waste', 'soil quality', 'national drug codes', 'steel industry', 'base load power', 'cloud analytics', 'transportation demand management', 'tls', 'elasticity', 'wholesale energy markets', 'hypothesis testing', '340b program', 'equal access to healthcare services', 'content generation', 'cpt codes', 'shipping and maritime industry', 'fits', 'sustainable innovation', 'exit strategy', 'behavioral analytics', 'supply chain transparency', 'run rate', 'demand response', 'ethical sourcing', 'value-based purchasing', 'entrepreneur', 'network security', 'organizational development', 'product launch', 'solar energy', 'risk assessment', 'energy justice', 'employment agreement', 'cybersecurity', 'carbon footprint', 'customer experience', 'leadership development', 'pfa', 'air pollution control', 'startup community', 'cms', 'energy policy act', 'nationally determined contributions', 'mro', 'cybersecurity awareness training', 'infectious disease control', 'cdc', 'compliance management', 'data analytics', 'air quality', 'global health', 'secure coding practices', 'business continuity planning', 'patient safety', 'video generation', 'nrf', 'crisis related to opioid misuse', 'transportation infrastructure financing', 'climate change indicators', 'cobra', 'product development', 'e.g., nist, iso/iec 27001', 'paas', 'startup incubator', 'nuclear energy', 'funding evaluation', 'clean power plan', 'trademark registration', 'msc', 'health economics', 'health insurance portability and accountability act', 'data lineage', 'endangered species', 'aircraft manufacturing', 'national institutes of health', 'anti-corruption policies', 'ethical ai', 'microgrids', 'sdgs', 'e.g., leed', 'power plants', 'european union aviation safety agency', 'talent acquisition', 'research impact', 'ocean conservation', 'social engineering defense', 'health information exchange', 'anti-bribery and corruption', 'decarbonization', 'bundled payments for care improvement', 'firewalls', 'startup competition', 'insider threat management', 'governance, risk, and compliance', 'policy development', 'demo day', 'glacier retreat', 'land use planning', 'classification', 'agile development', 'bridges', 'environmental law', 'ics', 'it procurement and acquisition', 'stt', 'data imputation', 'clean technology', 'socially responsible investing', 'clean energy technologies', 'carbon reduction projects', 'model interpretability', 'task force on climate-related financial disclosures', 'damage assessment', 'inland waterways', 'trade compliance', 'snf', 'cement industry', 'energy export', 'protected health information', 'cloud service level agreements', 'remote healthcare services', 'accountable care organization', 'collaboration networks', 'climate justice', 'airlines', 'venture capital', 'policy implementation', 'data aggregation', 'technology strategy', 'investor', 'climate change governance', 'gans', 'startup', 'change management', 'philanthropy', 'energy efficiency directive', 'fsc', 'food and beverage industry', 'high-speed rail', 'heavy industry', 'founder', 'extreme weather events', 'risk management', 'climate action', 'patent filing', 'grant reviews', 'climate change ethics']
project_2025_custom_taxonomy = ["custom_taxonomy", 'Biden Administration', 'Trump Administration', 'Critical Race Theory', 'Constitution', 'Market ', 'Money', 'Policy', 'Antitrust', 'Political', 'Supreme Court', 'Domestic', 'Security', 'Export', 'Import', 'Manufacturing', 'Census', 'Food', 'Immigration', 'Visa', 'Regulations', 'Courts', 'U.S. Constitution', 'Nuclear', 'China', 'Russia', 'Border', 'Military', 'Senate', 'Trade', 'Climate', 'Congress', 'Conversative', 'Wall', 'Liberal', 'Election', 'President', 'Republican', 'Democrat', 'Democratic', 'Pro-life', 'Transgender', 'Border', 'Economic', 'Defense', 'Budget', 'Deficit', 'Business', 'Artificial Intelligence', 'Technology', 'Gender', 'Children', 'Women', 'Family', 'Families', 'Marriage ', 'Equity', 'Progressive', 'Inclusion', 'LGBTQ', 'Fraud', 'Election', 'Legal', 'Government', 'Reagan', 'Legislation', 'Freedom', 'Trade', 'Tariffs ', 'Energy', 'Education', 'Housing', 'Civil Rights', 'Cybersecurity', 'Welfare', 'Jobs', 'Judicial ', 'Innovation', 'Enterprise', 'Federal', 'State', 'Minority', 'Party', 'Communist', 'Socialist ', 'Tax', 'Investment', 'Debt', 'Budget', 'Taxpayer', 'Cold War', 'European', 'Regulatory', 'Departments', 'Transition', 'Law', 'Media', 'Bureaucracy ', 'Media', 'Foreign ', 'Finance', 'Independent Regulatory Agencies', 'Executive Office', 'White House', 'Central Personnel Agencies', 'Department of Homeland Security', 'Defense', 'Department of State', 'Intelligence Community', 'Media Agencies', 'Exim', 'Agency for International Development', 'Department of Agriculture', 'Department of Education', 'Department of Energy', 'Environmental Protection Agency', 'Department of Health and Human Services', 'Department of Housing and Urban Development', 'Department of the Interior', 'Department of Justice', 'Department of Labor', 'Department of Transportation', 'Department of Veterans Affairs', 'Department of Commerce', 'Department of the Treasury ', 'Export Import Bank', 'Small Business Administration', 'Financial Regulatory Agencies ', 'Federal Communications Commission', 'Federal Election Commission', 'Federal Trade Commission']
peloton_custom_taxonomy = ["Peloton", "Team Sports", "Soccer", "Basketball", "American Football", "Baseball", "Hockey", "Volleyball", "Individual Sports", "Tennis", "Golf", "Boxing", "Wrestling", "Gymnastics", "Swimming", "Racquet Sports", "Badminton", "Squash", "Table Tennis", "Combat Sports", "Judo", "Karate", "Taekwondo", "Mixed Martial Arts", "MMA", "Track and Field", "Running", "Long Jump", "Pole Vault", "Shot Put", "Water Sports", "Rowing", "Sailing", "Surfing", "Diving", "Winter Sports", "Skiing", "Snowboarding", "Ice Skating", "Curling", "Indoor Cycling", "Live Classes", "On-Demand Classes", "Power Zone Training", "Treadmill", "Running Classes", "Walking Classes", "Bootcamp Classes", "Strength", "Upper Body", "Lower Body", "Full Body", "Yoga", "Vinyasa", "Power Yoga", "Restorative Yoga", "Meditation", "Sleep", "Stress Relief", "Focus", "Cardio", "HIIT", "Dance Cardio", "Shadowboxing", "Cardiovascular Training", "Running", "Cycling", "Rowing", "Strength Training", "Weightlifting", "Bodyweight Exercises", "Resistance Bands", "Flexibility Training", "Stretching", "Yoga", "Pilates", "Endurance Training", "Long-Distance Running", "Triathlons", "Marathon Training", "Functional Training", "CrossFit", "Bootcamp", "Kettlebell Workouts", "Specialized Training", "Sports-Specific Training", "Rehabilitation Exercises", "Prenatal and Postnatal Fitness", "Sports Coaching", "Team Coaching", "Individual Coaching", "Skill Development", "Fitness Coaching", "Personal Training", "Group Fitness Instruction", "Online Coaching", "Health Coaching", "Nutrition Coaching", "Wellness Coaching", "Behavioral Change Coaching", "Performance Coaching", "Mental Skills Coaching", "Motivational Coaching", "Recovery and Rehabilitation Coaching", "Youth Coaching", "Youth Sports", "Fitness for Kids", "Developmental Programs", "Executive and Life Coaching", "Leadership Coaching", "Career Coaching", "Life Skills Coaching"]
custom_taxonomy = ['startup', 'business', 'private equity', 'exit strategy', 'burn rate', 'team building', 'performance review', 'valuation cap', 'edtech', 'early-stage startup', 'customer segmentation', 'startup pitch', 'skills training', 'incorporation', 'customer acquisition', 'customer retention', 'trademark registration', 'business plan', 'ipo', 'marketing strategy', 'co-working space', 'market fit', 'venture funding', 'social media strategy', 'angel investor', 'user experience (ux)', 'pivot', 'business accelerator', 'fintech', 'non-disclosure agreement (nda)', 'product launch', 'patent filing', 'leadership', 'equity', 'startup valuation', 'mezzanine financing', 'startup accelerator', 'business model', 'corporation', 'infrastructure as a service (iaas)', 'break-even point', 'late-stage startup', 'proptech', 'software as a service (saas)', 'market disruption', 'run rate', 'market analysis', 'networking events', 'regulatory compliance', 'cap table', 'due diligence', 'management', 'capital raise', 'startup mentor', 'series a/b/c funding', 'scaling', 'minimal viable product (mvp)', 'venture capitalist', 'user acquisition', 'revenue model', 'series c funding', 'term sheet', 'entrepreneurship', 'startup law', 'startup office', 'startup ecosystem', 'series d funding', 'prototype', 'patent', 'venture capital', 'series b funding', 'startup advisor', 'advisor', 'profit margin', 'family and friends round', 'remote work', 'funding round', 'accelerator', 'value proposition', 'board member', 'sales strategy', 'equity stake', 'debt financing', 'platform as a service (paas)', 'startup incubator', 'product development', 'tech stack', 'y combinator', 'proof of concept', 'pitch deck', 'vesting schedule', 'disruptive technology', 'limited partner', 'mvp (minimum viable product)', 'startup competition', 'series a funding', 'mvp', 'bootstrapping', 'staff development', 'venture partner', 'angel round', 'employee hiring', 'sweat equity', 'venture round', 'shareholder agreement', 'seed capital', 'financial model', 'incubator', 'contract negotiation', 'healthtech', 'data privacy', 'pitch competition', 'entrepreneur', 'convertible note', 'innovation', 'growth capital', 'pre-seed funding', 'crowdfunding', 'shares', 'seed round', 'stock options', 'unique selling proposition (usp)', 'competitive analysis', 'traction', 'limited liability company (llc)', 'talent acquisition', 'intellectual property (ip)', 'revenue-based financing', 'startup culture', 'equity financing', 'digital marketing', 'industry conference', 'employment agreement', 'fundraising', 'corporate governance', 'seed funding', 'growth hacking', 'venture debt', 'trademark', 'angel funding', 'partnership', 'business development', 'founder', 'minimum viable product', 'series e funding', 'serial entrepreneur', 'general partner', 'demo day', 'intellectual property rights', 'startup community', 'lean startup', 'go-to-market strategy', 'organizational culture', 'techstars', 'greentech', 'user interface (ui)', 'equity crowdfunding', 'engineer', 'scientist', 'investor']

"""## Flair Code Block"""

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
  entity_text = entity_text.replace("'s", "")
  entity_text = entity_text.replace("'", "")
  entity_text = entity_text.replace('"', '')

  return entity_text, entity_type

common_first_names = ["aaden", "aadhya", "aadit", "aadvik", "aadya", "aahana", "aaiden", "aaleyah", "aaliyah", "aanya", "aarav", "aariz", "aarna", "aarohi", "aaron", "aarush", "aarya", "aaryan", "aavya", "aayan", "aayden", "abbey", "abbie", "abbigail", "abby", "abbygail", "abdel", "abdiel", "abdul", "abdullah", "abdulrahman", "abel", "abelardo", "abhiram", "abigail", "abigale", "abner", "abraham", "abram", "abriella", "abrielle", "abril", "abygail", "acacia", "ace", "achilles", "ada", "adalee", "adalie", "adalina", "adalind", "adaline", "adaly", "adalyn", "adalyne", "adalynn", "adalynne", "adam", "adamaris", "adan", "addalyn", "addelyn", "addilyn", "addilynn", "addison", "addisyn", "addyson", "adela", "adelaida", "adelaide", "adele", "adelia", "adelin", "adelina", "adeline", "adelle", "adelyn", "adelyne", "adelynn", "adelynne", "aden", "adhvik", "adhya", "adiel", "adilene", "adina", "adira", "aditi", "aditya", "adler", "adley", "adolfo", "adonai", "adonis", "adrian", "adriana", "adrianna", "adriel", "adrien", "adrienne", "advait", "advaith", "advik", "advika", "aedan", "aella", "aeris", "aero", "agam", "agnes", "agustin", "ahaan", "ahana", "ahmad", "ahmed", "aida", "aidan", "aiden", "aila", "ailani", "ailany", "ailee", "aileen", "ailyn", "aimee", "ainhoa", "ainsley", "aira", "aisha", "aislinn", "aislynn", "aitana", "aiyana", "aiyanna", "aiza", "aizen", "aj", "ajay", "akash", "akhil", "akira", "akiva", "aksel", "akshara", "alaa", "alaia", "alaina", "alan", "alana", "alanah", "alani", "alanna", "alannah", "alaric", "alaya", "alayah", "alayla", "alayna", "alba", "albert", "alberto", "aldair", "alden", "aldo", "aleah", "alec", "aleeah", "aleen", "aleena", "aleenah", "aleia", "alejandra", "alejandro", "alek", "aleksander", "aleksandra", "alena", "alessa", "alessandra", "alessandro", "alessia", "alessio", "alex", "alexa", "alexander", "alexandra", "alexandria", "alexandro", "alexey", "alexi", "alexia", "alexis", "alexxa", "alexzander", "aleyah", "aleyda", "aleyna", "aleyza", "alfonso", "alfred", "alfredo", "ali", "alia", "aliah", "aliana", "alianna", "alice", "alicia", "alijah", "alina", "alinah", "alinna", "alisa", "alisha", "alison", "alissa", "alisson", "alistair", "alivia", "aliya", "aliyah", "aliyana", "aliza", "alizae", "alizah", "alizay", "alize", "allan", "allen", "allie", "allison", "allisson", "ally", "allyson", "alma", "alondra", "alonso", "alonzo", "alora", "alp", "alston", "althea", "alton", "alvaro", "alvin", "alya", "alyah", "alyana", "alyanna", "alycia", "alyna", "alynna", "alyson", "alyssa", "alysson", "amadeus", "amado", "amaia", "amairani", "amairany", "amal", "amalia", "amanda", "amani", "amar", "amara", "amarah", "amare", "amari", "amariah", "amarie", "amaris", "amaryllis", "amaya", "amayah", "amaziah", "ambar", "amber", "amberly", "ambrose", "ameer", "ameera", "amelia", "amelie", "america", "amerie", "amethyst", "amia", "amiah", "amie", "amilia", "amin", "amina", "aminah", "amir", "amira", "amirah", "amiya", "amiyah", "amor", "amora", "amos", "amy", "an", "ana", "anabel", "anabella", "anabelle", "anahi", "anahit", "anais", "anaisha", "anaiya", "anaiyah", "anakin", "analeah", "analee", "anali", "analia", "analiah", "analisa", "analise", "analiyah", "analy", "ananya", "anastasia", "anay", "anaya", "anayah", "ander", "anders", "anderson", "andi", "andie", "andre", "andrea", "andreas", "andrei", "andres", "andrew", "andrey", "andy", "anessa", "angad", "angel", "angela", "angelia", "angelica", "angelie", "angelina", "angeline", "angelique", "angelo", "angie", "ani", "ania", "anika", "anisa", "anisha", "anissa", "anita", "aniya", "aniyah", "anjali", "ann", "anna", "annabel", "annabell", "annabella", "annabelle", "annabeth", "annaleah", "annalee", "annalia", "annalie", "annaliese", "annalisa", "annalise", "annamarie", "anne", "annelise", "annette", "annie", "annika", "anny", "ansel", "ansh", "anson", "anthony", "antoine", "antoinette", "anton", "antonella", "antonia", "antonio", "antony", "anuhea", "anvi", "anvika", "anya", "aoife", "apollo", "april", "ara", "arabella", "arabelle", "araceli", "aracely", "aram", "arantxa", "arantza", "aranza", "araya", "archer", "archie", "arden", "areg", "areli", "arely", "aren", "ares", "arham", "ari", "aria", "ariadna", "ariadne", "ariah", "arian", "ariana", "arianna", "arianny", "ariany", "aribella", "arie", "ariel", "ariela", "ariella", "arielle", "aries", "arin", "arina", "aris", "arisbeth", "arius", "ariya", "ariyah", "arizbeth", "arjan", "arjun", "arleen", "arlene", "arlet", "arleth", "arlett", "arlette", "arley", "arlo", "armaan", "arman", "armando", "armani", "armen", "armin", "arnav", "arnold", "arnulfo", "aron", "arrow", "arsalan", "arsen", "artem", "artemis", "arthur", "arturo", "arvin", "arwen", "arya", "aryan", "aryana", "aryanna", "asa", "asees", "ash", "asha", "ashanti", "asher", "ashley", "ashly", "ashlyn", "ashlynn", "ashton", "asia", "asiya", "aslan", "asma", "aspen", "asra", "aston", "astrid", "atarah", "athan", "atharv", "athena", "atlas", "atom", "atreyu", "atticus", "atziri", "aubree", "aubrey", "aubri", "aubriana", "aubrianna", "aubrie", "aubriella", "aubrielle", "audra", "audree", "audrey", "audriana", "audrianna", "audrie", "audrina", "august", "augustin", "augustine", "augustus", "aulani", "aura", "aurelia", "aurelio", "aurora", "austen", "austin", "austyn", "autumn", "ava", "avah", "avalon", "avalyn", "avalynn", "avan", "avani", "aveline", "aven", "averi", "averie", "avery", "avi", "aviana", "avianna", "avigail", "aviva", "avleen", "avneet", "avni", "avonlea", "avril", "avyaan", "avyan", "avyanna", "axel", "axl", "axton", "aya", "ayaan", "ayaansh", "ayah", "ayan", "ayana", "ayanna", "ayansh", "aydan", "ayden", "aydin", "ayesha", "ayla", "aylan", "aylani", "ayleen", "aylen", "aylin", "ayra", "ayush", "ayva", "ayvah", "azael", "azaiah", "azalea", "azaria", "azariah", "azeneth", "aziel", "azlan", "azrael", "azriel", "azucena", "azul", "baby", "bailee", "bailey", "baker", "baltazar", "barbara", "baron", "barrett", "barron", "barry", "bastian", "baylee", "baylor", "bear", "beatrice", "beatrix", "beatriz", "beau", "beck", "beckett", "beckham", "belen", "belinda", "bella", "bellamy", "bellarose", "belle", "ben", "benaiah", "benedict", "benicio", "benito", "benjamin", "bennett", "benny", "benson", "bentley", "benton", "berenice", "berkeley", "berlin", "bernadette", "bernard", "bernardo", "bernice", "betania", "bethany", "betsy", "betty", "beverly", "bexley", "bianca", "bianka", "bilal", "bill", "billie", "billy", "birdie", "bishop", "bjorn", "blaine", "blair", "blaire", "blaise", "blake", "blakely", "blanca", "blayke", "blaze", "blessing", "blythe", "bo", "boaz", "bobbie", "bobby", "bode", "boden", "bodhi", "bodi", "bodie", "bonnie", "boone", "boris", "borna", "boston", "bowen", "bowie", "boyd", "braden", "bradley", "brady", "bradyn", "braeden", "braelyn", "braelynn", "braiden", "branden", "brandon", "branson", "brantley", "braulio", "braxton", "brayan", "brayden", "braydon", "braylen", "braylon", "brea", "breanna", "brecken", "bree", "brenda", "brendan", "brenden", "brendon", "brenna", "brennan", "brent", "brentley", "brenton", "brett", "bria", "brian", "briana", "brianna", "briar", "bridget", "bridgette", "briella", "brielle", "briggs", "brighton", "brigitte", "brinley", "brisa", "briseida", "briseis", "briseyda", "brissa", "bristol", "britney", "brittany", "brittney", "brixton", "brock", "broderick", "brodie", "brody", "bronson", "brooke", "brooklyn", "brooklynn", "brooks", "bruce", "bruno", "bryan", "bryanna", "bryant", "bryce", "brycen", "bryden", "brylee", "brynlee", "brynn", "brysen", "bryson", "byron", "cade", "caden", "cadence", "caelan", "caesar", "caia", "caiden", "cailyn", "cain", "cairo", "caitlin", "caitlyn", "cal", "caleb", "cali", "calia", "calista", "caliyah", "calla", "callahan", "callan", "callen", "callie", "calliope", "callum", "calvin", "cam", "cambria", "camden", "camdyn", "camellia", "cameron", "camila", "camilla", "camille", "camilo", "campbell", "camryn", "canaan", "candace", "candice", "candy", "cannon", "capri", "cara", "carina", "carissa", "carl", "carla", "carlie", "carlo", "carlos", "carly", "carmela", "carmella", "carmelo", "carmen", "carol", "carolina", "caroline", "carolyn", "carrie", "carson", "carter", "case", "casen", "casey", "cash", "cason", "casper", "caspian", "cassandra", "cassia", "cassian", "cassidy", "cassie", "cassius", "castiel", "cataleya", "catalina", "caterina", "catherine", "cathy", "cattaleya", "cattleya", "cayden", "caydence", "cayla", "caylee", "cayson", "cecelia", "cecilia", "cecily", "cedar", "cedric", "celeste", "celia", "celina", "celine", "cesar", "chad", "chance", "chandler", "chanel", "chanelle", "channing", "charbel", "charis", "charity", "charlee", "charleigh", "charlene", "charles", "charley", "charli", "charlie", "charlize", "charlotte", "chase", "chaya", "chelsea", "chelsey", "cherish", "chevelle", "cheyenne", "chiara", "chloe", "chris", "christian", "christiana", "christina", "christine", "christopher", "christy", "cian", "ciara", "cielo", "cienna", "cillian", "cindy", "citlali", "citlalli", "citlaly", "claire", "clara", "clare", "clarence", "clarissa", "clark", "clarke", "claudia", "clay", "clayton", "clementine", "cleo", "clifford", "clint", "clive", "clover", "clyde", "coco", "cody", "coen", "cohen", "colbie", "colby", "cole", "coleman", "colette", "colin", "colleen", "collette", "collin", "collins", "colson", "colt", "colten", "colton", "conner", "connie", "connor", "conor", "conrad", "constance", "constantine", "cooper", "cora", "coral", "coralie", "coraline", "corbin", "cordelia", "corey", "corina", "corinna", "corinne", "cormac", "cortez", "cory", "cosette", "courtney", "craig", "crew", "cristal", "cristian", "cristiano", "cristina", "cristo", "cristobal", "cristofer", "cristopher", "crosby", "cru", "cruz", "crystal", "cullen", "curren", "curtis", "cynthia", "cyrus", "daenerys", "dafne", "dahlia", "dailyn", "daira", "daisy", "dakota", "dalary", "dale", "daleyza", "dalia", "daliah", "dalila", "dalilah", "dallas", "dalton", "damari", "damaris", "damian", "damien", "damon", "dan", "dana", "danae", "dane", "danelly", "d'angelo", "dani", "dania", "danica", "daniel", "daniela", "daniella", "danielle", "danika", "danilo", "danna", "danny", "dante", "daphne", "dara", "darby", "darcy", "daria", "darian", "dariana", "dariel", "darien", "dario", "darius", "darla", "darleen", "darlene", "darlyn", "daron", "darrell", "darren", "darryl", "darwin", "darya", "dash", "dasha", "dashiell", "dave", "davian", "david", "davin", "davina", "davion", "davis", "davit", "davon", "dawn", "dawson", "dax", "daxon", "daxton", "dayana", "dayanara", "dayanna", "daylan", "daylen", "dayra", "dayton", "deacon", "dean", "deandre", "de'andre", "deanna", "debora", "deborah", "decker", "declan", "deegan", "deklan", "delaney", "delanie", "delia", "delilah", "della", "delylah", "demetrius", "demi", "denali", "denim", "denis", "denise", "denisse", "deniz", "dennis", "deon", "dereck", "derek", "derick", "derrick", "deshawn", "desiree", "desmond", "destinee", "destiny", "dev", "devan", "devansh", "deven", "devin", "devina", "devon", "devyn", "dexter", "dezmond", "dhruv", "diamond", "diana", "diane", "dianna", "diego", "diesel", "dilan", "dillon", "dimitri", "dina", "dion", "dior", "divine", "dixie", "diya", "dolores", "domingo", "dominic", "dominick", "dominik", "dominique", "don", "donald", "donna", "donnie", "donovan", "donte", "dora", "dorian", "doris", "dorothy", "douglas", "drake", "draven", "dream", "drew", "duke", "dulce", "duncan", "dustin", "dwayne", "dylan", "eamon", "ean", "earl", "eason", "easton", "eddie", "eddy", "eden", "eder", "edgar", "edie", "edison", "edith", "edmund", "edric", "edson", "eduardo", "edward", "edwin", "eesa", "efrain", "efren", "egypt", "eiden", "eila", "eileen", "eimy", "eisley", "eitan", "eithan", "eiza", "ela", "elaina", "elaine", "elan", "elara", "elayna", "eleanor", "eleanora", "eleazar", "elena", "eleni", "eli", "elia", "eliah", "eliam", "elian", "eliana", "elianna", "elias", "eliel", "eliette", "eliezer", "elif", "eligh", "elih", "elijah", "elin", "elina", "elinor", "elio", "eliot", "eliott", "elisa", "elisabeth", "elise", "eliseo", "elisha", "elissa", "eliyanah", "eliza", "elizabella", "elizabeth", "elizah", "ella", "elle", "ellen", "ellery", "ellia", "elliana", "ellianna", "ellie", "elliot", "elliott", "ellis", "ellison", "elly", "elodie", "eloisa", "eloise", "elon", "elora", "elowen", "eloy", "elsa", "elsie", "elvin", "elvira", "elvis", "elyana", "elyas", "elyse", "elysia", "elyssa", "ema", "emani", "emanuel", "emarie", "ember", "emberly", "emelia", "emelie", "emeline", "emely", "emerald", "emeri", "emerie", "emerson", "emersyn", "emery", "emi", "emiko", "emil", "emile", "emilee", "emilia", "emiliana", "emiliano", "emilie", "emilio", "emily", "emir", "emma", "emmalee", "emmaline", "emmalyn", "emmalynn", "emmanuel", "emmanuelle", "emmarie", "emmarose", "emme", "emmeline", "emmet", "emmett", "emmi", "emmie", "emmitt", "emmy", "emory", "emrie", "emry", "ender", "enoch", "enrique", "ensley", "enzo", "ephraim", "eric", "erica", "erick", "ericka", "erik", "erika", "erin", "eris", "ernest", "ernesto", "eros", "ervin", "erwin", "eryn", "esai", "esdras", "esme", "esmeralda", "esperanza", "esteban", "estefania", "estefany", "estela", "estella", "estelle", "ester", "estevan", "esther", "estrella", "ethan", "ethen", "etta", "eugene", "eunice", "eva", "evalina", "evalyn", "evalynn", "evan", "evander", "evangelina", "evangeline", "eve", "evelin", "evelina", "evelyn", "evelynn", "ever", "everardo", "everest", "everett", "everlee", "everleigh", "everley", "everly", "evette", "evie", "evolet", "ewan", "ezekiel", "ezequiel", "eziah", "ezio", "ezra", "ezrah", "fabian", "fabiola", "fahad", "faith", "fallon", "farah", "farhan", "faris", "farrah", "fateh", "fatima", "faye", "felicity", "felipe", "felix", "fernanda", "fernando", "fidel", "finley", "finn", "finnegan", "finnian", "finnick", "finnley", "fiona", "fiorella", "fisher", "flavio", "fletcher", "flor", "flora", "florence", "flynn", "ford", "forest", "forrest", "fox", "frances", "francesca", "francesco", "francine", "francis", "francisco", "franco", "frank", "frankie", "franklin", "fred", "freddie", "freddy", "frederick", "fredrick", "fredy", "freya", "freyja", "frida", "gabriel", "gabriela", "gabriella", "gabrielle", "gael", "gage", "gaia", "gala", "galen", "galilea", "galileah", "gareth", "garrett", "garrison", "gary", "gauge", "gavin", "gema", "gemma", "genaro", "gene", "genesis", "geneva", "genevie", "genevieve", "geoffrey", "george", "georgia", "georgina", "geovanni", "gerald", "geraldine", "gerard", "gerardo", "german", "gerson", "gia", "giada", "gian", "giana", "giancarlo", "gianluca", "gianna", "gianni", "giavanna", "gibran", "gibson", "gideon", "gigi", "gilbert", "gilberto", "gina", "gino", "gio", "giovani", "giovanna", "giovanni", "giovanny", "gisela", "gisele", "giselle", "gisselle", "giulia", "giuliana", "giulianna", "giulietta", "gizelle", "gladys", "glenn", "gloria", "glory", "goldie", "gonzalo", "gordon", "grace", "gracelyn", "gracelynn", "gracie", "graciela", "grady", "graham", "grant", "gray", "graysen", "grayson", "grecia", "gregorio", "gregory", "greta", "grettel", "grey", "greysen", "greyson", "griffin", "griselda", "guadalupe", "guillermo", "guinevere", "gunnar", "gunner", "gurfateh", "gurnoor", "gustavo", "guy", "gwen", "gwendolyn", "gwenyth", "gwyneth", "hadassah", "hadi", "hadley", "haiden", "hailee", "hailey", "hailie", "hakeem", "haley", "halle", "hallie", "halo", "hamza", "hana", "haniya", "hank", "hanna", "hannah", "hans", "hanson", "harlan", "harlee", "harleen", "harlem", "harley", "harlie", "harlow", "harlowe", "harlyn", "harmony", "harold", "harper", "harriet", "harris", "harrison", "harry", "harvey", "hasan", "hassan", "hattie", "havana", "haven", "hawa", "hawk", "haya", "hayden", "hayes", "hayk", "haylee", "hayley", "haylie", "hazel", "heath", "heather", "heaven", "heavenly", "hector", "heidi", "heidy", "helen", "helena", "hellen", "hendrick", "hendrix", "henley", "hennessy", "henri", "henrik", "henry", "heriberto", "hernan", "hezekiah", "hiram", "hiro", "holden", "holland", "hollis", "holly", "honesty", "honor", "hope", "horacio", "hosanna", "houston", "howard", "hudson", "hugh", "hugo", "humberto", "hunter", "husna", "hussein", "huxley", "ian", "ibrahim", "ida", "idris", "ignacio", "iker", "ila", "ilan", "ilana", "ileana", "ilene", "iliana", "illiana", "ilona", "ilyas", "iman", "imani", "immanuel", "imogen", "imran", "inaaya", "inara", "inaya", "inayah", "india", "indiana", "indie", "indigo", "ines", "inez", "ingrid", "ira", "ireland", "irelynn", "irene", "irie", "iris", "irlanda", "irma", "irvin", "irving", "isa", "isaac", "isaak", "isabel", "isabela", "isabell", "isabella", "isabelle", "isadora", "isai", "isaiah", "isaias", "isela", "isha", "ishaan", "ishan", "ishika", "ishmael", "isiah", "isidro", "isla", "ismael", "ismail", "isobel", "israel", "issa", "issac", "italia", "italy", "itzae", "itzamara", "itzayana", "itzel", "itzia", "iva", "ivaan", "ivan", "ivana", "ivanka", "ivanna", "ivette", "ivory", "ivy", "iyla", "izaac", "izaak", "izabel", "izabella", "izabelle", "izael", "izaiah", "izan", "izayah", "izel", "izrael", "jace", "jacey", "jaciel", "jack", "jackeline", "jackie", "jacklyn", "jacklynn", "jackson", "jaclyn", "jacob", "jacoby", "jacqueline", "jacquelyn", "jad", "jada", "jade", "jaden", "jadiel", "jadon", "jaeden", "jael", "jaelyn", "jaelynn", "jagger", "jahir", "jai", "jaida", "jaiden", "jailyne", "jaime", "jair", "jairo", "jake", "jakob", "jalen", "jaliyah", "jamal", "jamari", "james", "jameson", "jamie", "jamila", "jamileth", "jamir", "jamison", "jan", "jana", "janae", "jane", "janella", "janelle", "janelly", "janessa", "janet", "janeth", "janice", "janie", "janine", "janiyah", "janna", "janney", "jaqueline", "jared", "jaretzy", "jarrett", "jarvis", "jase", "jasiah", "jasiel", "jasleen", "jaslene", "jaslyn", "jasmin", "jasmine", "jason", "jasper", "javier", "jax", "jaxen", "jaxon", "jaxson", "jaxton", "jaxx", "jaxxon", "jay", "jayce", "jaycee", "jayceon", "jaycie", "jaycob", "jayda", "jayde", "jayden", "jaydon", "jayla", "jaylah", "jaylani", "jaylee", "jayleen", "jaylen", "jaylene", "jaylie", "jaylin", "jayline", "jaylyn", "jaylynn", "jayne", "jayson", "jazelle", "jaziah", "jaziel", "jazleen", "jazlene", "jazlyn", "jazlynn", "jazmin", "jazmine", "jazmyn", "jazzlyn", "jean", "jeanette", "jedidiah", "jeff", "jefferson", "jeffery", "jeffrey", "jelena", "jemma", "jencarlos", "jenelle", "jenesis", "jenevieve", "jenna", "jennie", "jennifer", "jenny", "jensen", "jenson", "jeremiah", "jeremias", "jeremih", "jeremy", "jericho", "jerimiah", "jermaine", "jerome", "jerry", "jeshua", "jesiah", "jeslyn", "jessa", "jesse", "jessi", "jessiah", "jessica", "jessie", "jesslyn", "jesus", "jet", "jethro", "jett", "jewel", "jhene", "jia", "jianna", "jillian", "jimena", "jimmy", "jiovanni", "jiraiya", "jiselle", "jiya", "joan", "joana", "joanna", "joanne", "joaquin", "jocelyn", "jocelyne", "jocelynn", "joe", "joel", "joelle", "joey", "johan", "johana", "johann", "johanna", "john", "johnathan", "johnnie", "johnny", "joie", "jolee", "jolene", "jolie", "jon", "jonah", "jonas", "jonatan", "jonathan", "jonathon", "jones", "jordan", "jordi", "jordy", "jordyn", "jordynn", "jorge", "jose", "josefina", "joselin", "joseluis", "joselyn", "joseph", "josephina", "josephine", "josh", "joshua", "josiah", "josias", "josie", "joslyn", "josselyn", "josslyn", "josue", "journee", "journey", "jovan", "jovani", "jovanni", "jovanny", "jovie", "joy", "joyce", "joziah", "jrue", "juan", "juanita", "jubilee", "judah", "jude", "judith", "judy", "juelz", "jules", "julia", "julian", "juliana", "julianna", "julianne", "julie", "julien", "juliet", "julieta", "julieth", "julietta", "juliette", "julio", "julissa", "julius", "jullian", "jun", "juna", "june", "junior", "juniper", "juno", "jurnee", "justice", "justin", "justine", "justus", "kabir", "kace", "kacey", "kade", "kaden", "kadence", "kaeden", "kaelani", "kaelyn", "kaelynn", "kahlil", "kai", "kaia", "kaiden", "kaidence", "kaila", "kailah", "kailani", "kailee", "kailey", "kaily", "kailyn", "kain", "kaine", "kainoa", "kaira", "kairi", "kairo", "kaison", "kaitlyn", "kaitlynn", "kaiya", "kalani", "kalea", "kaleah", "kaleb", "kalel", "kal-el", "kalena", "kaleo", "kaley", "kali", "kalia", "kaliah", "kalila", "kalina", "kaliyah", "kallen", "kallie", "kalvin", "kamari", "kamden", "kameron", "kamila", "kamilah", "kamilla", "kamille", "kamryn", "kane", "kannon", "kanoa", "kara", "karam", "kareem", "karely", "karen", "karim", "karina", "karis", "karisma", "karissa", "karl", "karla", "karlee", "karley", "karlie", "karma", "karolina", "karoline", "karson", "karsyn", "karter", "kasen", "kasey", "kash", "kashton", "kason", "kassandra", "kassidy", "kataleya", "katalina", "katarina", "kate", "katelyn", "katelynn", "katerina", "katharine", "katherine", "kathleen", "kathryn", "kathy", "katia", "katie", "katrina", "katy", "katya", "kavin", "kavya", "kay", "kaya", "kaycee", "kayden", "kaydence", "kayla", "kaylah", "kaylani", "kaylee", "kayleen", "kayleigh", "kaylen", "kaylene", "kayley", "kayli", "kaylie", "kaylin", "kaylynn", "kaysen", "kayson", "keanu", "keaton", "keegan", "keenan", "kehlani", "keila", "keilani", "keily", "keira", "keiry", "keith", "kekoa", "kellan", "kellen", "kelly", "kelsea", "kelsey", "kelsie", "kelvin", "ken", "kendall", "kendra", "kendrick", "kenia", "kenji", "kenley", "kenna", "kennedi", "kennedy", "kenneth", "kenny", "kensington", "kensley", "kent", "kenya", "kenzie", "kenzo", "keoni", "keshav", "keven", "kevin", "keyla", "keziah", "khadija", "khai", "khaleesi", "khalia", "khalid", "khalil", "khang", "khari", "khloe", "khristian", "kiaan", "kian", "kiana", "kianna", "kiara", "kiera", "kieran", "kiernan", "kiley", "kilian", "killian", "kimber", "kimberly", "kimora", "king", "kingsley", "kingston", "kinley", "kinsey", "kinslee", "kinsley", "kinzley", "kion", "kira", "kiran", "kirra", "kirsten", "klarissa", "klayton", "knox", "koa", "kobe", "koda", "kody", "koen", "kohen", "kole", "kolton", "konnor", "kora", "korbin", "kori", "korra", "kris", "krish", "krisha", "krishna", "kristen", "kristian", "kristin", "kristina", "kristine", "kristopher", "kristy", "kruz", "krystal", "kurt", "kurtis", "kya", "kyan", "kyara", "kye", "kyla", "kylah", "kylan", "kyle", "kylee", "kyler", "kylie", "kylo", "kynlee", "kyra", "kyran", "kyree", "kyrie", "kyro", "kyson", "lacey", "lachlan", "lady", "laila", "lailah", "lailani", "lainey", "laith", "lake", "lamar", "lana", "lance", "landen", "landon", "landyn", "lane", "laney", "langston", "lani", "laniyah", "lara", "lareina", "larissa", "larry", "laura", "laurel", "lauren", "laurence", "lauryn", "lawrence", "lawson", "laya", "layah", "layan", "layla", "laylah", "laylani", "layne", "layton", "lazaro", "lazarus", "lea", "leah", "leandro", "leanna", "leanne", "lee", "leela", "leen", "leena", "legacy", "legend", "leia", "leif", "leighton", "leila", "leilah", "leilani", "leilanie", "leilany", "lela", "leland", "lena", "lenin", "lennon", "lennox", "lenny", "leo", "leon", "leona", "leonard", "leonardo", "leonel", "leonidas", "leonie", "leopold", "leopoldo", "leroy", "lesley", "leslie", "lesly", "lester", "leticia", "lev", "levi", "levon", "levy", "lewis", "lex", "lexa", "lexi", "lexie", "leyla", "leylah", "leylani", "lia", "liah", "liam", "lian", "liana", "lianna", "libby", "liberty", "lidia", "liev", "lila", "lilah", "lili", "lilia", "lilian", "liliana", "lilianna", "lilit", "lilith", "lillian", "lilliana", "lillianna", "lillie", "lillith", "lilly", "lillyana", "lillyanna", "lily", "lilyana", "lilyanna", "lina", "lincoln", "linda", "lindsay", "lindsey", "link", "linkin", "linnea", "lino", "linus", "lionel", "liora", "lisa", "lisandro", "lisette", "litzy", "liv", "livia", "liya", "liyana", "liz", "liza", "lizbeth", "lizeth", "lizette", "lloyd", "lluvia", "lochlan", "logan", "lois", "loki", "lola", "london", "londyn", "lorelai", "lorelei", "lorena", "lorenzo", "loretta", "lori", "lorraine", "lotus", "louie", "louis", "louisa", "louise", "lourdes", "love", "lovely", "loyalty", "luc", "luca", "lucas", "lucca", "lucero", "lucia", "lucian", "luciana", "lucianna", "luciano", "lucie", "lucien", "lucille", "lucinda", "lucio", "lucius", "lucky", "lucy", "luella", "luis", "luisa", "luka", "lukas", "luke", "lula", "lulu", "luna", "lupita", "luther", "lux", "luz", "lyanna", "lydia", "lyla", "lylah", "lyle", "lyndon", "lynette", "lynn", "lyra", "lyric", "mabel", "mac", "maceo", "macey", "maci", "macie", "mack", "mackenzie", "macy", "madalyn", "madalynn", "madden", "maddie", "maddison", "maddix", "maddox", "maddux", "madeleine", "madeline", "madelyn", "madelyne", "madelynn", "madilyn", "madilynn", "madison", "madisyn", "madyson", "mae", "maeve", "magali", "magaly", "magdalena", "magdalene", "maggie", "magnolia", "magnus", "maha", "mai", "maia", "mailen", "maira", "maisie", "maison", "maite", "maiya", "majesty", "major", "makai", "makaio", "makayla", "makena", "makenna", "makenzie", "makoa", "maksim", "malachi", "malak", "malakai", "malakhi", "malaki", "malani", "malaya", "malayah", "malaysia", "malcolm", "maleah", "malek", "malena", "malia", "maliah", "malik", "malina", "maliya", "maliyah", "mallory", "mandy", "maneh", "mannat", "manolo", "manuel", "mara", "marbella", "marc", "marcel", "marcela", "marceline", "marcelino", "marcella", "marcello", "marcellus", "marcelo", "marco", "marcos", "marcus", "marely", "maren", "margaret", "margarita", "margaux", "margo", "margot", "mari", "maria", "mariah", "mariam", "marian", "mariana", "marianna", "marianne", "mariano", "maribel", "maribelle", "maricela", "marie", "mariel", "mariela", "mariella", "marielle", "marigold", "marilyn", "marina", "mario", "marion", "marisa", "marisol", "marissa", "maritza", "mariyah", "marjorie", "mark", "markus", "marla", "marlee", "marleigh", "marlene", "marley", "marlie", "marlon", "marlow", "marlowe", "mars", "marshall", "martha", "martin", "marvin", "marwa", "mary", "maryam", "maryjane", "masen", "mason", "massimo", "mateo", "matheo", "mathew", "mathias", "mathis", "matias", "matilda", "matix", "matt", "matteo", "matthew", "matthias", "mattias", "mattie", "mattis", "maurice", "mauricio", "mauro", "maven", "maverick", "mavis", "mavrick", "max", "maxim", "maximilian", "maximiliano", "maximillian", "maximo", "maximus", "maxine", "maxton", "maxwell", "maxx", "may", "maya", "mayah", "mayeli", "mayleen", "maylin", "mayra", "mayson", "mayte", "mazie", "mazzy", "mccoy", "mckayla", "mckenna", "mckenzie", "mckinley", "meadow", "meera", "megan", "meghan", "mehar", "mehtab", "mei", "meilani", "mekhi", "melani", "melania", "melanie", "melannie", "melany", "melia", "melina", "melinda", "melisa", "melissa", "melodie", "melody", "melvin", "memphis", "mercedes", "mercy", "meredith", "merida", "merrick", "messiah", "metztli", "mia", "miabella", "miah", "micaela", "micah", "micaiah", "michael", "michaela", "micheal", "michelangelo", "michelle", "miguel", "mika", "mikael", "mikaela", "mikah", "mikayla", "mike", "mikel", "mikhail", "mila", "milagro", "milagros", "milah", "milan", "milana", "milani", "milania", "mileena", "milena", "miles", "miley", "miliana", "milla", "miller", "millicent", "millie", "milly", "milo", "milton", "mina", "minerva", "minh", "mira", "mirabella", "mirabelle", "miracle", "miranda", "mireya", "miriam", "misael", "misha", "mitchell", "miya", "miyah", "moana", "mohamed", "mohammad", "mohammed", "moira", "moises", "mollie", "molly", "mona", "monica", "monique", "monroe", "monserrat", "monserrath", "monte", "montgomery", "montserrat", "morgan", "moriah", "morrison", "moses", "moshe", "muhammad", "murphy", "musa", "mustafa", "mya", "myah", "myla", "mylah", "mylene", "myles", "mylo", "myra", "myron", "nabeel", "nadia", "nadine", "nahla", "nahomi", "nahomy", "nahum", "naia", "naila", "nailah", "naima", "nala", "nalani", "naliyah", "nancy", "naomi", "naomy", "nara", "narek", "nash", "nasir", "natalee", "natalia", "natalie", "nataly", "natalya", "natanael", "natasha", "nate", "nathalia", "nathalie", "nathaly", "nathan", "nathanael", "nathaniel", "nathen", "naveah", "naveen", "navya", "naya", "nayeli", "nayla", "neal", "neel", "neev", "nefertari", "nehemiah", "nehemias", "neil", "nellie", "nelly", "nelson", "neo", "nestor", "nevaeh", "neveah", "neymar", "nia", "niam", "nicholas", "nichole", "nick", "nickolas", "nico", "nicolai", "nicolas", "nicole", "nicolette", "nigel", "nihal", "nika", "nikhil", "nikita", "nikki", "nikko", "niklaus", "niko", "nikolai", "nikolas", "nila", "nina", "nirvaan", "nirvana", "nivaan", "nixon", "niya", "niyah", "noa", "noah", "noam", "noble", "noe", "noel", "noelani", "noelia", "noelle", "noemi", "nohemi", "nola", "nolan", "noor", "noora", "nora", "norah", "norma", "norman", "nour", "nova", "novah", "novalee", "nya", "nyah", "nyjah", "nyla", "nylah", "nyomi", "nyra", "oakley", "obadiah", "ocean", "octavia", "octavio", "oden", "odessa", "odette", "odin", "ofelia", "olin", "olive", "oliver", "olivia", "ollie", "olympia", "om", "omar", "omari", "onyx", "oona", "opal", "ophelia", "oren", "ori", "oriana", "orion", "orlando", "oscar", "oseias", "osiel", "osiris", "oskar", "osmar", "osvaldo", "oswaldo", "otis", "otto", "owen", "oziel", "ozzie", "ozzy", "pablo", "paige", "paislee", "paisley", "paityn", "paloma", "pamela", "paola", "paolo", "paris", "parker", "pascual", "patricia", "patricio", "patrick", "paul", "paula", "paulette", "paulina", "pauline", "paulo", "pax", "paxton", "payton", "pearl", "pedro", "penelope", "penny", "pepper", "perla", "persephone", "peter", "petra", "peyton", "pharaoh", "philip", "phillip", "phoebe", "phoenix", "pia", "pierce", "pierre", "pilar", "piper", "polina", "poppy", "porter", "pranav", "precious", "preslee", "presley", "preston", "prince", "princess", "princeton", "priscila", "priscilla", "prisha", "priya", "queen", "queena", "queenie", "quentin", "quincy", "quinn", "quinten", "quintin", "quinton", "rachael", "rachel", "rae", "raegan", "raelyn", "raelynn", "rafael", "raiden", "rain", "raina", "raine", "raj", "ralph", "rami", "ramiro", "ramon", "ramona", "ramses", "ramsey", "randall", "randy", "rania", "ranveer", "raphael", "raquel", "rashad", "raul", "raven", "ray", "raya", "rayan", "rayden", "raylan", "rayleen", "raylene", "raylynn", "raymond", "raymundo", "rayna", "rayne", "rayyan", "reagan", "rebeca", "rebecca", "rebekah", "reece", "reed", "reef", "reese", "regan", "regina", "reginald", "reid", "reign", "reina", "remi", "remington", "remmy", "remy", "ren", "renata", "renato", "rene", "renee", "renesmee", "renzo", "reuben", "rex", "rey", "reya", "reyaan", "reyan", "reyansh", "reyes", "reyna", "reynaldo", "rhea", "rhett", "rhiannon", "rhys", "ria", "riaan", "rian", "ricardo", "richard", "richie", "rick", "ricky", "rico", "ridge", "rigoberto", "rihaan", "rihanna", "riley", "rilynn", "rina", "rio", "ripley", "rishaan", "rishi", "rita", "ritchie", "river", "riya", "robbie", "robert", "roberto", "robin", "robyn", "rocco", "rocio", "rockwell", "rocky", "roderick", "rodney", "rodolfo", "rodrigo", "roen", "rogelio", "roger", "rogue", "rohan", "roland", "rolando", "roman", "rome", "romeo", "romi", "romina", "romy", "ron", "ronald", "ronaldo", "ronan", "ronen", "ronin", "ronnie", "rory", "rosa", "rosabella", "rosalee", "rosalia", "rosalie", "rosalina", "rosalind", "rosalinda", "rosalyn", "rosalynn", "rosario", "rose", "roselyn", "roselynn", "rosemarie", "rosemary", "rosie", "roslyn", "rosy", "rowan", "rowen", "rowyn", "roxana", "roxanna", "roxanne", "roy", "roya", "royal", "royalty", "royce", "ruben", "rubi", "ruby", "rudra", "rudy", "ruhi", "rumi", "russell", "ruth", "ryan", "ryann", "ryatt", "ryden", "ryder", "ryker", "rylan", "ryland", "rylee", "ryleigh", "rylen", "rylie", "rylin", "ryu", "saanvi", "sabella", "sabina", "sabine", "sabrina", "sade", "sadie", "safa", "sage", "sahana", "sahara", "sahasra", "sahib", "sai", "said", "saif", "saige", "sailor", "saint", "saira", "saisha", "sakura", "saleh", "salem", "sally", "salma", "salome", "salvador", "salvatore", "sam", "sama", "samaira", "samantha", "samanvi", "samar", "samara", "samarth", "samaya", "sami", "samir", "samira", "samiyah", "sammy", "samson", "samuel", "sana", "sanaya", "sandra", "sandy", "saniyah", "santana", "santhiago", "santiago", "santino", "santos", "sanvi", "saoirse", "sapphire", "sara", "sarah", "sarahi", "sarai", "saray", "sariah", "sarina", "sariyah", "sasha", "saul", "savanna", "savannah", "savina", "sawyer", "saylor", "sayuri", "scarlet", "scarlett", "scarlette", "scott", "scout", "seamus", "sean", "sebastian", "sebastien", "sedona", "seerat", "sehaj", "selah", "selena", "selene", "selim", "selina", "selma", "senna", "sequoia", "serafina", "seraphina", "serena", "serene", "serenity", "sergey", "sergio", "serina", "seth", "seven", "sevyn", "shaan", "shaelyn", "shaila", "shaina", "shanaya", "shane", "shannon", "shant", "sharon", "shaun", "shaurya", "shawn", "shay", "shayan", "shayla", "shayna", "sheila", "shelby", "sheldon", "shepard", "shepherd", "sherlyn", "sherry", "shiloh", "shira", "shirley", "shiv", "shivansh", "shoshana", "shreya", "shriya", "shyla", "sia", "siana", "siddharth", "sidney", "siena", "sienna", "sierra", "silas", "silvia", "simon", "simone", "simran", "sincere", "sione", "sir", "sire", "sirena", "sitara", "siya", "skarlett", "sky", "skye", "skyla", "skylah", "skylar", "skyler", "slade", "slater", "sloan", "sloane", "smith", "sofia", "sofie", "soham", "sol", "solana", "soleil", "solomon", "sonia", "sonja", "sonny", "sonora", "sonya", "sophia", "sophie", "sora", "soraya", "soren", "sparrow", "spencer", "sri", "stacey", "stacy", "stanley", "star", "starla", "steele", "stefan", "stefano", "stella", "stellan", "stephan", "stephanie", "stephany", "stephen", "sterling", "stetson", "steve", "steven", "stevie", "stone", "stormy", "subhan", "suhani", "sulaiman", "sulayman", "sullivan", "sultan", "summer", "sunny", "suri", "susan", "susana", "susanna", "susie", "sutton", "swara", "sydney", "syed", "sylas", "sylvia", "sylvie", "symphony", "syrus", "tabitha", "tadeo", "taha", "tahlia", "taj", "tala", "talia", "taliyah", "tallulah", "talon", "talya", "tamara", "tamia", "tammy", "tania", "tanner", "tanvi", "tanya", "tara", "taryn", "tate", "tatiana", "tatum", "tatyana", "taya", "tayden", "taylor", "teagan", "teegan", "tegan", "temperance", "tenley", "teo", "terence", "teresa", "terrance", "terrell", "terrence", "terry", "tesla", "tess", "tessa", "tevita", "thaddeus", "thalia", "thatcher", "thea", "theo", "theodora", "theodore", "theresa", "thiago", "thomas", "thor", "thorin", "tia", "tiago", "tiana", "tianna", "tiara", "tiberius", "tiffany", "tim", "timothy", "tina", "tinley", "tinsley", "titan", "titus", "tobias", "toby", "tom", "tomas", "tommy", "toni", "tony", "tori", "torin", "trace", "tracy", "travis", "trent", "trenton", "trevor", "trey", "trinity", "trisha", "tristan", "tristen", "triston", "troy", "tru", "trystan", "tucker", "turner", "ty", "tyler", "tyrone", "tyson", "ulices", "ulises", "ulysses", "uma", "unique", "uriah", "uriel", "urijah", "uziel", "uzziah", "vada", "valen", "valencia", "valente", "valentin", "valentina", "valentine", "valentino", "valeria", "valerie", "valery", "valor", "van", "vance", "vanellope", "vanessa", "vania", "vanity", "vanna", "vanya", "varun", "vaughn", "vayda", "ved", "veda", "vedanth", "veer", "venice", "venus", "vera", "verity", "veronica", "viaan", "vianey", "vianna", "vianney", "vic", "vicente", "vicky", "victor", "victoria", "vida", "vidal", "vienna", "viggo", "vihaan", "vikram", "viktor", "viktoria", "vince", "vincent", "vincenzo", "viola", "violet", "violeta", "violett", "violetta", "violette", "viraaj", "viraj", "virgil", "virginia", "viridiana", "virlan", "vishnu", "vito", "vivaan", "vivian", "viviana", "vivianna", "vivianne", "vivien", "vivienne", "vladimir", "von", "wade", "walker", "wallace", "walter", "warren", "watson", "waverly", "waylon", "wayne", "wednesday", "wells", "wendy", "wes", "wesley", "wesson", "west", "westin", "westley", "weston", "westyn", "whitney", "wilder", "wiley", "wilhelmina", "will", "willa", "william", "willie", "willow", "wilson", "winnie", "winston", "winter", "wolf", "wolfgang", "wren", "wyatt", "wylder", "wylie", "wynter", "xaiden", "xander", "xavi", "xavier", "xena", "ximena", "ximenna", "xiomara", "xitlali", "xitlaly", "xochitl", "xzavier", "yadiel", "yadira", "yael", "yahaira", "yahir", "yahya", "yair", "yamila", "yamilet", "yamileth", "yana", "yandel", "yaneli", "yara", "yareli", "yarely", "yaretzi", "yaretzy", "yaritza", "yash", "yasiel", "yasmeen", "yasmin", "yasmine", "yatzil", "yatziri", "yazmin", "yesenia", "yeshua", "yi", "yichen", "yoel", "yohan", "yolanda", "yosef", "yoselin", "yoselyn", "yousef", "yousif", "youssef", "yousuf", "ysabella", "yu", "yuliana", "yulissa", "yuna", "yunus", "yurani", "yuri", "yuritzi", "yusra", "yusuf", "yuvaan", "yuvan", "yuvraj", "yvette", "yvonne", "zabdiel", "zac", "zach", "zachariah", "zacharias", "zachary", "zack", "zackary", "zackery", "zadie", "zahara", "zahir", "zahra", "zaid", "zaiden", "zain", "zaina", "zainab", "zaira", "zaire", "zakai", "zakaria", "zamira", "zander", "zane", "zaniyah", "zara", "zarah", "zaria", "zariah", "zariyah", "zavier", "zaya", "zayd", "zayda", "zayden", "zayla", "zaylee", "zayn", "zayna", "zaynab", "zayne", "zayra", "zechariah", "zeina", "zeke", "zelda", "zella", "zena", "zendaya", "zephaniah", "zephyr", "zeppelin", "zeus", "zev", "zi", "zia", "zinnia", "zion", "zipporah", "ziva", "ziya", "zoe", "zoey", "zoie", "zola", "zooey", "zora", "zorawar", "zoya", "zulema", "zuleyka", "zuri", "zyaire", "zyana", "zyanya", "zyla"]
def is_entity_garbage(entity_text, entity_type):
    length = len(entity_text)
    alnum_count = sum(c.isalnum() for c in entity_text)
    space_count = entity_text.count(' ')

    if entity_type == 'PERSON' and entity_text.lower() in common_first_names:
        return True

    # strip entity_text off any leading or trailing whitespaces
    entity_text = entity_text.strip()

    # if entity_text has a dot (.) at the end or start
    if entity_text.endswith('.') or entity_text.startswith('.'):
        return True
    
    # if entity_text ends with a dot followed by any character
    if re.search(r'\.\w', entity_text) or re.search(r'\.\w\w', entity_text) or re.search(r'\.\w\w\w', entity_text) or re.search(r'\.\w\w\w\w', entity_text):
        return True
    
    if length > 30 and space_count < 1:
        return True
    
    if length > 30:
        # if any of the words in entity_text is more than 30 characters long, return True
        if any(len(word) > 30 for word in entity_text.split()):
            return True
    
    # Special character check
    # Optional: You might want to adjust this to remove entities with unusual characters
    if re.search(r'[^a-zA-Z0-9\s\.\-\'\,]', entity_text) and entity_type != 'MONEY' and entity_type != 'PERCENT' and entity_type != 'QUANTITY' and entity_type != 'LAW':
        return True
    
    # Custom rule for specific patterns (e.g., excessive numbers or unusual patterns)
    if re.search(r'\d{3,}', entity_text) and entity_type != 'MONEY' and entity_type != 'PERCENT' and entity_type != 'QUANTITY' and entity_type != 'LAW':
        return True
    
    return False

def for_ingestion_pipeline(single_line_paragraph):
  entities, entity_text_only, entity_type_only = [], [], []
  cannonical_map = {}
  global custom_taxonomy, entity_blacklist

  # remove any /n or /t characters from the single_line_paragraph
  single_line_paragraph = single_line_paragraph.replace("\\n", " ").replace("\\t", " ")
  # remove forward and backward slashes
  single_line_paragraph = single_line_paragraph.replace("\\", " ").replace('/', " ")
  
  # get entities from Vanilla Flair (ner-ontonotes-large model)
  sentence = Sentence(single_line_paragraph)
  tagger.predict(sentence)
  for entity in sentence.get_spans('ner'):
    entity_text_temp = entity.text
    entity_type_temp = entity.labels[0].value
    confidence_temp = entity.labels[0].score

    # if entity_text_temp is 'cardinal' or 'ordinal', skip it
    if entity_type_temp == 'CARDINAL' or entity_type_temp == 'ORDINAL' or entity_type_temp == 'DATE' or entity_type_temp == 'TIME' or entity_type_temp == 'MONEY' or entity_type_temp == 'PERCENT' or entity_type_temp == 'QUANTITY':
      continue
    
    # if entity_text_temp matches any of the enitity_blacklist items, skip it
    if entity_text_temp.lower() in entity_blacklist:
        print(f"Entity: {entity_text_temp} is in the entity_blacklist. Skipping...")
        continue

    if is_entity_garbage(entity_text_temp, entity_type_temp):
        if  entity_type_temp == 'PERSON':
            entity_type_temp = 'GARBAGE_NAME'
        else:
            entity_type_temp = 'GARBAGE'

    if confidence_temp < 0.99:
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

  results = programatic_taxonomy_detection(single_line_paragraph, custom_taxonomy)

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


def fuzzy_positive_pairs(all_entities, threshold=70):
    call_stack = []
    for entity in tqdm(all_entities, desc="Finding redundant entities..."):
        # remove entity from all_entities and save it in a temp variable
        temp_all_entities = all_entities.copy()
        temp_all_entities.remove(entity)
        for other_entity in temp_all_entities:
            if fuzz.ratio(entity, other_entity) >= threshold:
                call_stack.append((entity, other_entity))

    return call_stack

def fuzzy_positive_pairs_combination(entities_batch_one, entities_batch_two, threshold=70):
    call_stack = []
    for entity in tqdm(entities_batch_one, desc="Finding redundant entities..."):
        for other_entity in entities_batch_two:
            if fuzz.ratio(entity, other_entity) >= threshold:
                call_stack.append((entity, other_entity))

    return call_stack

## Redundant Entity Map implementation
def optimize_redundant_entities(entities, redundant_entity_mapping, garbage_entities, all_entities, ignore_entity_types, input_source='ingestion', debug_level=0):
    # replace the redundant entities in the redundant_entity_mapping of all the entities accordingly
    # for entity in tqdm(entities, desc="Optimizing redundant entities NER_tags, NER_tags_with_type and NER_cannonical_map..."):
    # for entity in entities:
    for entity in tqdm(entities, desc="Optimizing redundant entities..."):
        # focus on NER_tags
        if input_source == 'ingestion':
            NER_tags_as_list_of_strings = entity.metadata["NER_tags"]
        elif input_source == 'kg':
            NER_tags_as_list_of_strings = entity["metadata"]["NER_tags"]
        else:
            print(f"Invalid input source: {input_source}")
            return

        if debug_level >= 4:
            print(f"\nstage 1 - NER_tags: {NER_tags_as_list_of_strings}")
        
        # if NER_tags_as_list_of_strings is a string, convert it to a list
        try:
            if type(NER_tags_as_list_of_strings) is str:
                # convert the NER_tags into a list of strings
                NER_tags_as_list_of_strings = NER_tags_as_list_of_strings.strip("[").strip("]") # remove the starting and ending square brackets, if present
                NER_tags_as_list_of_strings = NER_tags_as_list_of_strings.replace("'s", "")  # remove single quotes followed by an 's'
                NER_tags_as_list_of_strings = NER_tags_as_list_of_strings.replace("'", "")  # remove single quotes
                NER_tags_as_list_of_strings = NER_tags_as_list_of_strings.replace("\"", "")  # remove double quotes
                NER_tags_as_list_of_strings = NER_tags_as_list_of_strings.split(", ")
        except Exception as e:
            print(f"Error in converting NER_tags_as_list_of_strings to a list: {e}")
            print(f"NER_tags_as_list_of_strings: {NER_tags_as_list_of_strings}")
            print(f"type(NER_tags_as_list_of_strings): {type(NER_tags_as_list_of_strings)}")
            return

        if debug_level >= 4:
            print(f"stage 2 - NER_tags: {NER_tags_as_list_of_strings}")
        
        updated_NER_tags = NER_tags_as_list_of_strings.copy()

        for current_tag_base in NER_tags_as_list_of_strings:
            for non_redundant_entity, redundants in redundant_entity_mapping.items():
                for redundant_entity in redundants:
                    if current_tag_base == redundant_entity and current_tag_base in all_entities: # 2nd condition: make sure that the entity is present in all_entities, effectively ignoring the entity types in ignore_entity_types
                        if non_redundant_entity not in updated_NER_tags:
                            updated_NER_tags.append(non_redundant_entity)
                        if redundant_entity in updated_NER_tags:
                            updated_NER_tags.remove(redundant_entity)
                        for garbage_entity in garbage_entities:
                            if garbage_entity in updated_NER_tags:
                                updated_NER_tags.remove(garbage_entity)

        if debug_level >= 4:
            print(f"stage 3 - NER_tags: {updated_NER_tags}")

        updated_NER_tags = list(set(updated_NER_tags)) # Ensure uniqueness

        if input_source == 'ingestion':
            entity.metadata["NER_tags"] = updated_NER_tags
        elif input_source == 'kg':
            entity["metadata"]["NER_tags"] = updated_NER_tags
        else:
            print(f"Invalid input source: {input_source}")
            return

        if debug_level >= 4:
            if input_source == 'ingestion':
                print(f"stage 4 - NER_tags: {entity.metadata['NER_tags']}")
            elif input_source == 'kg':
                print(f"stage 5 - NER_tags: {entity['metadata']['NER_tags']}")
            else:
                print(f"Invalid input source: {input_source}")
                return entities

        # focus on NER_tags_with_type
        if input_source == 'ingestion':
            tags_with_type = entity.metadata["NER_tags_with_type"]
        elif input_source == 'kg':
            tags_with_type = entity["metadata"]["NER_tags_with_type"]
        else:
            print(f"Invalid input source: {input_source}")
            return
        
        if debug_level >= 4:
            print(f"\nstage 1 - tags_with_type: {tags_with_type}")

        # if tags_with_type is a string, convert it to a list
        try:
            if type(tags_with_type) is str:
                # convert the NER_tags into a list of strings
                tags_with_type = tags_with_type.strip("[").strip("]") # remove the starting and ending square brackets, if present
                tags_with_type = tags_with_type.replace("'", "")  # remove single quotes
                tags_with_type = tags_with_type.replace("\"", "")  # remove double quotes
                tags_with_type = tags_with_type.split(", ")
        except Exception as e:
            print(f"Error in converting tags_with_type to a list: {e}")
            print(f"tags_with_type: {tags_with_type}")
            print(f"type(tags_with_type): {type(tags_with_type)}")
            return
        
        updated_tags_with_type = tags_with_type.copy()
        
        for tag in tags_with_type:
            current_tag_base = tag.split(" (")[0]
            if current_tag_base in all_entities: # make sure that the entity is present in all_entities, effectively ignoring the entity types in ignore_entity_types
                for non_redundant_entity, redundants in redundant_entity_mapping.items():
                    if any(current_tag_base == red.split(" (")[0] for red in redundants):
                        updated_tag = non_redundant_entity + tag[tag.find("(")-1:]  # Preserve type
                        if updated_tag not in updated_tags_with_type:
                            updated_tags_with_type.append(updated_tag)
                        if tag in updated_tags_with_type:
                            updated_tags_with_type.remove(tag)
                        for garbage_entity in garbage_entities:
                            if garbage_entity in updated_tags_with_type:
                                updated_tags_with_type.remove(garbage_entity)

        updated_tags_with_type = list(set(updated_tags_with_type))

        if debug_level >= 4:
            print(f"stage 3 - tags_with_type: {updated_tags_with_type}")
        
        if input_source == 'ingestion':
            entity.metadata["NER_tags_with_type"] = updated_tags_with_type
        elif input_source == 'kg':
            entity["metadata"]["NER_tags_with_type"] = updated_tags_with_type
        else:
            print(f"Invalid input source: {input_source}")
            return

        # focus on NER_cannonical_map
        if input_source == 'ingestion':
            temp = entity.metadata["NER_cannonical_map"]
        elif input_source == 'kg':
            temp = entity["metadata"]["NER_cannonical_map"]
        else:
            print(f"Invalid input source: {input_source}")
            return
        
        if debug_level >= 4:
            print(f"\nstage 1 - NER_cannonical_map: {temp}")
        
        if type(temp) is str:
            temp = json.loads(temp)
        
        if debug_level >= 4:
            print(f"stage 2 - NER_cannonical_map: {temp}")
        
        relevant_entity_types = []
        for entity_type in temp.keys():
            if entity_type not in ignore_entity_types:
                relevant_entity_types.append(entity_type)
        for entity_type in relevant_entity_types:
            try:
                for non_redundant_entity, redundant_entities in redundant_entity_mapping.items():
                    if debug_level >= 5:
                        print(f"\nredundant_entity: {redundant_entity}")
                        print(f"temp[entity_type]: {temp[entity_type]}")
                        print(f"entity_type: {entity_type}")
                    for redundant_entity in redundant_entities:
                        if redundant_entity in temp[entity_type]:
                            if debug_level >= 5:
                                print(f"\nRemoving {redundant_entity} from temp[{entity_type}], and adding {non_redundant_entity}")

                            if redundant_entity in temp[entity_type]:
                                temp[entity_type].remove(redundant_entity)

                            if non_redundant_entity not in temp[entity_type]:
                                temp[entity_type].append(non_redundant_entity)
                        for garbage_entity in garbage_entities:
                            if garbage_entity in temp[entity_type]:
                                temp[entity_type].remove(garbage_entity)
                
            except Exception as e:
                print(f"\nError in processing NER_cannonical_map: {e}")   
                print(f"entity_type: {entity_type}")
                print(f"type(entity_type): {type(entity_type)}")
                if input_source == 'ingestion':
                    print(f"entity.metadata[\"NER_cannonical_map\"]: {entity.metadata['NER_cannonical_map']}")
                    print(f"type(entity.metadata[\"NER_cannonical_map\"]): {type(entity.metadata['NER_cannonical_map'])}")
                elif input_source == 'kg':
                    print(f"entity[\"metadata\"][\"NER_cannonical_map\"]: {entity['metadata']['NER_cannonical_map']}")
                    print(f"type(entity[\"metadata\"][\"NER_cannonical_map\"]): {type(entity['metadata']['NER_cannonical_map'])}")
                else:
                    print(f"Invalid input source: {input_source}")
                    return
                print(f"temp: {temp}")
                print(f"type(temp): {type(temp)}")
                return

        try:
            if input_source == 'ingestion':
                entity.metadata["NER_cannonical_map"] = temp
            elif input_source == 'kg':
                entity["metadata"]["NER_cannonical_map"] = temp
            else:
                print(f"Invalid input source: {input_source}")
                return
        except Exception as e:
            print(f"Error in updating NER_cannonical_map: {e}")
            print(f"entity_type: {entity_type}")
            print(f"type(entity_type): {type(entity_type)}")
            if input_source == 'ingestion':
                print(f"entity.metadata[\"NER_cannonical_map\"]: {entity.metadata['NER_cannonical_map']}")
                print(f"type(entity.metadata[\"NER_cannonical_map\"]): {type(entity.metadata['NER_cannonical_map'])}")
            elif input_source == 'kg':
                print(f"entity[\"metadata\"][\"NER_cannonical_map\"]: {entity['metadata']['NER_cannonical_map']}")
                print(f"type(entity[\"metadata\"][\"NER_cannonical_map\"]): {type(entity['metadata']['NER_cannonical_map'])}")
            else:
                print(f"Invalid input source: {input_source}")
                return
            print(f"temp: {temp}")
            print(f"type(temp): {type(temp)}")
            return
        
        if debug_level >= 4:
            if input_source == 'ingestion':
                print(f"stage 3 - NER_cannonical_map: {entity.metadata['NER_cannonical_map']}")
            elif input_source == 'kg':
                print(f"stage 3 - NER_cannonical_map: {entity['metadata']['NER_cannonical_map']}")
            else:
                print(f"Invalid input source: {input_source}")
                return
            
    return entities


"""## Flask Code Block"""

app = Flask(__name__)

@app.route('/', methods=["POST"])
def hello():
  # data = request.get_json()
  # text = data['text']
  # import environment variable 'WORKER' from the shell
  num_processes_for_ngrok = os.getenv('WORKERS')
  num_processes_for_ngrok = int(num_processes_for_ngrok)
  recognized_entities = {"warning": "This is not an active/production endpoint. Production enpoints include 'ingestion_pipeline' and 'doccano_pre_annotation'","num_processes_for_ngrok":num_processes_for_ngrok}
  recognized_entities = jsonify(recognized_entities) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response
  return recognized_entities

@app.route('/ingestion_pipeline', methods=["POST"])
def ingestion_pipeline():
  data = request.get_json()
  text = data['text']
  try:
    recognized_entities = for_ingestion_pipeline(text) # for ingestion pipeline
  except Exception as e:
    recognized_entities = {"error": str(e), "Entities": [f'error: {str(e)}'], "EntityTextOnly": [f'error: {str(e)}'], "EntityTypeOnly": [f'error: {str(e)}'], "CannonicalMap": {"error": str(e)}}
    print(str(e))
  recognized_entities = jsonify(recognized_entities) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response

  return recognized_entities

@app.route('/err', methods=["POST"])
def err():
  data = request.get_json()
  text = data['text']
  # non_redundant_entity = classify_entities(text)
  non_redundant_entity = classify_entities_and_return_parameters(text)
  non_redundant_entity = jsonify(non_redundant_entity) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response
  return non_redundant_entity

@app.route('/err_batch', methods=["POST"])
def err_batch():
  data = request.get_json()
  text = data['text']
  non_redundant_entity = classify_entities_and_return_parameters_batch(text)
  non_redundant_entity = jsonify(non_redundant_entity) # essential because returning a dictionary directly from a Flask route does not automatically convert it to a JSON response
  return non_redundant_entity

@app.route('/optimize_redundant_entities', methods=["POST"])
def optimize_redundant_entities_call():
  data = request.get_json()
  entities = data['entities']
  redundant_entity_mapping = data['redundant_entity_mapping']
  garbage_entities = data['garbage_entities']
  all_entities = data['all_entities']
  ignore_entity_types = data['ignore_entity_types']
  input_source = data['input_source']
  debug_level = data['debug_level']
  try:
    entities = optimize_redundant_entities(entities, redundant_entity_mapping, garbage_entities, all_entities, ignore_entity_types, input_source, debug_level)
  except Exception as e:
    print(str(e))
    entities.append({"error": str(e)})

  entities = jsonify(entities)
  return entities

@app.route('/fuzzy_batch', methods=["POST"])
def fuzzy_batch():
    data = request.get_json()
    all_entities, threshold = data['text'], data['threshold']
    fuzzy_positive_pairs_batch = fuzzy_positive_pairs(all_entities, threshold)
    fuzzy_positive_pairs_batch = jsonify(fuzzy_positive_pairs_batch)
    return fuzzy_positive_pairs_batch

@app.route('/fuzzy_batch_combination', methods=["POST"])
def fuzzy_batch_combination():
    data = request.get_json()
    entities_batch_one, entities_batch_two, threshold = data['entities_batch_one'], data['entities_batch_two'], data['threshold']
    fuzzy_positive_pairs_batch = fuzzy_positive_pairs_combination(entities_batch_one, entities_batch_two, threshold)
    fuzzy_positive_pairs_batch = jsonify(fuzzy_positive_pairs_batch)
    return fuzzy_positive_pairs_batch