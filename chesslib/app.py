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

custom_taxonomies = ["startup_taxonomy", "ifc_taxonomies_taxonomy", "energy_taxonomy", "federal_health_taxonomy", "disaster_management_taxonomy", "climate_taxonomy", "transportation_taxonomy", "environment_taxonomy", "aviation_taxonomy", "data_analytics_taxonomy", "generative_ai_taxonomy", "cyber_security_taxonomy", "regulatory_policy_taxonomy", "enterprise_cloud_solutions_taxonomy", "research_evaluation_taxonomy", "advisory_taxonomy", "public_health_taxonomy"]

custom_taxonomy_name = ['startup', 'business incubator', 'co-founder', 'Liberty Mutual Fire Insurance Company', 'equity crowdfunding']
startup_taxonomy = ['startup', 'business', 'private equity', 'exit strategy', 'burn rate', 'team building', 'performance review', 'valuation cap', 'edtech', 'early-stage startup', 'customer segmentation', 'startup pitch', 'skills training', 'incorporation', 'customer acquisition', 'customer retention', 'trademark registration', 'business plan', 'ipo', 'marketing strategy', 'co-working space', 'market fit', 'venture funding', 'social media strategy', 'angel investor', 'user experience (ux)', 'pivot', 'business accelerator', 'fintech', 'non-disclosure agreement (nda)', 'product launch', 'patent filing', 'leadership', 'equity', 'startup valuation', 'mezzanine financing', 'startup accelerator', 'business model', 'corporation', 'infrastructure as a service (iaas)', 'break-even point', 'late-stage startup', 'proptech', 'software as a service (saas)', 'market disruption', 'run rate', 'market analysis', 'networking events', 'regulatory compliance', 'cap table', 'due diligence', 'management', 'capital raise', 'startup mentor', 'series a/b/c funding', 'scaling', 'minimal viable product (mvp)', 'venture capitalist', 'user acquisition', 'revenue model', 'series c funding', 'term sheet', 'entrepreneurship', 'startup law', 'startup office', 'startup ecosystem', 'series d funding', 'prototype', 'patent', 'venture capital', 'series b funding', 'startup advisor', 'advisor', 'profit margin', 'family and friends round', 'remote work', 'funding round', 'accelerator', 'value proposition', 'board member', 'sales strategy', 'equity stake', 'debt financing', 'platform as a service (paas)', 'startup incubator', 'product development', 'tech stack', 'y combinator', 'proof of concept', 'pitch deck', 'vesting schedule', 'disruptive technology', 'limited partner', 'mvp (minimum viable product)', 'startup competition', 'series a funding', 'mvp', 'bootstrapping', 'staff development', 'venture partner', 'angel round', 'employee hiring', 'sweat equity', 'venture round', 'shareholder agreement', 'seed capital', 'financial model', 'incubator', 'contract negotiation', 'healthtech', 'data privacy', 'pitch competition', 'entrepreneur', 'convertible note', 'innovation', 'growth capital', 'pre-seed funding', 'crowdfunding', 'shares', 'seed round', 'stock options', 'unique selling proposition (usp)', 'competitive analysis', 'traction', 'limited liability company (llc)', 'talent acquisition', 'intellectual property (ip)', 'revenue-based financing', 'startup culture', 'equity financing', 'digital marketing', 'industry conference', 'employment agreement', 'fundraising', 'corporate governance', 'seed funding', 'growth hacking', 'venture debt', 'trademark', 'angel funding', 'partnership', 'business development', 'founder', 'minimum viable product', 'series e funding', 'serial entrepreneur', 'general partner', 'demo day', 'intellectual property rights', 'startup community', 'lean startup', 'go-to-market strategy', 'organizational culture', 'techstars', 'greentech', 'user interface (ui)', 'equity crowdfunding', 'engineer', 'scientist', 'investor']
ifc_taxonomies_taxonomy = ["ifc taxonomies", "federal it modernization", "cloud computing", "virtualization", "data center consolidation", "cybersecurity", "agile development", "devops", "containerization", "microservices", "blockchain technology", "artificial intelligence", "ai", "machine learning", "robotic process automation", "rpa", "internet of things", "iot", "big data analytics", "data governance", "data integration", "open data", "geospatial data", "api management", "mobile application development", "user-centered design", "enterprise architecture", "it infrastructure optimization", "software defined networking", "sdn", "edge computing", "quantum computing", "5g network integration", "it service management", "itsm", "identity and access management", "iam", "continuous monitoring", "incident response", "risk management framework", "rmf", "governance, risk, and compliance", "grc", "it procurement and acquisition", "legacy system modernization", "digital transformation", "citizen-centric services", "open source software adoption", "secure coding practices", "performance engineering", "scalability and elasticity", "disaster recovery planning", "knowledge management", "change management", "it budgeting and cost management", "vendor management", "business process reengineering", "regulatory compliance", "it skills development", "sustainability in it operations"]
energy_taxonomy = ["energy", "renewable energy", "solar power", "wind energy", "hydropower", "biomass energy", "geothermal energy", "energy efficiency", "smart grids", "energy storage", "battery technologies", "electric vehicles", "evs", "grid modernization", "distributed energy resources", "ders", "microgrids", "energy management systems", "ems", "demand response", "energy conservation", "energy audits", "energy policy", "energy economics", "energy markets", "energy trading", "power generation", "natural gas", "oil and gas exploration", "shale gas", "lng", "liquefied natural gas", "coal and coal technologies", "nuclear power", "carbon capture and storage", "ccs", "carbon pricing", "energy transition", "energy infrastructure", "energy security", "energy independence", "energy resilience", "energy access", "renewable portfolio standards", "rps", "energy regulatory framework", "energy consumption patterns", "energy demand forecasting", "energy supply chain", "energy innovation", "energy financing", "energy risk management", "energy governance", "energy diplomacy", "energy justice", "sustainable energy", "energy education and awareness"]
federal_health_taxonomy = ["federal health", "clinical modification", "cpt codes", "current procedural terminology", "hcpcs codes", "healthcare common procedure coding system", "drgs", "diagnosis-related groups", "ndc codes", "national drug codes", "medicare part a/b/c/d", "medicare program parts", "hipaa", "health insurance portability and accountability act", "hhs", "department of health and human services", "fda", "food and drug administration", "cdc", "centers for disease control and prevention", "nih", "national institutes of health", "cms", "centers for medicare & medicaid services", "pqrs", "physician quality reporting system", "ehr", "electronic health records", "phi", "protected health information", "hitech act", "health information technology for economic and clinical health act", "macra", "medicare access and chip reauthorization act", "aco", "accountable care organization", "bpci", "bundled payments for care improvement", "fqhc", "federally qualified health center", "pdp", "prescription drug plan", "under medicare part d", "hmo", "health maintenance organization", "ppo", "preferred provider organization", "medicaid waivers", "various medicaid waiver programs", "chip", "children's health insurance program", "vbp", "value-based purchasing", "mips", "merit-based incentive payment system", "qpp", "quality payment program", "340b program", "drug pricing program", "cobra", "consolidated omnibus budget reconciliation act", "health insurance continuation", "ihs", "indian health service", "va health system", "veterans affairs health services", "snf", "skilled nursing facility", "home health services", "medicare-covered home health services", "mental health parity act", "legislation related to mental health coverage", "long-term care", "various services and facilities", "telehealth", "remote healthcare services", "ambulatory surgical centers", "outpatient surgical facilities", "health disparities", "differences in health outcomes among populations", "health literacy", "understanding of health information", "population health", "health outcomes of a group of individuals", "social determinants of health", "economic and social factors influencing health", "health equity", "equal access to healthcare services", "health information exchange", "sharing of electronic health information", "opioid epidemic", "crisis related to opioid misuse", "healthcare fraud", "illegal practices in the healthcare system", "patient safety", "measures to prevent harm in healthcare settings", "public health emergency preparedness", "readiness for health emergencies", "healthcare quality measures", "standards to assess healthcare performance", "healthcare financing", "funding mechanisms for healthcare services"]
disaster_management_taxonomy = ["disaster management", "natural disasters", "man-made disasters", "hazards", "risk assessment", "vulnerability", "resilience", "emergency response", "disaster recovery", "disaster mitigation", "incident command system", "ics", "emergency operations center", "eoc", "national response framework", "nrf", "disaster declaration", "evacuation", "shelter management", "mass care", "mass casualty incident", "mci", "search and rescue", "sar", "damage assessment", "critical infrastructure", "public health emergency", "medical surge", "triage", "psychological first aid", "pfa", "disaster communications", "community resilience", "social vulnerability", "climate change adaptation", "technological hazards", "radiological emergency", "chemical spills", "biological threats", "wildfire management", "floodplain management", "earthquake preparedness", "hurricane response", "tornado preparedness", "drought contingency", "infrastructure resilience", "risk communication", "early warning systems", "gis", "geographic information systems) in disaster management", "crisis mapping", "emergency alert system", "eas", "volunteer management", "donations management", "reconstruction", "financial assistance programs", "business continuity planning", "lessons learned"]
climate_taxonomy = ["climate", "greenhouse gases", "carbon footprint", "climate change adaptation", "climate change mitigation", "renewable energy", "energy efficiency", "carbon pricing", "carbon markets", "emissions trading", "carbon offsets", "climate policy", "paris agreement", "unfccc", "united nations framework convention on climate change", "ipcc", "intergovernmental panel on climate change", "global warming", "sea level rise", "ocean acidification", "deforestation", "afforestation", "carbon sequestration", "climate resilience", "climate justice", "adaptation financing", "mitigation strategies", "climate models", "extreme weather events", "heat waves", "drought", "floods", "glacier retreat", "arctic amplification", "carbon capture and storage", "ccs", "bioenergy", "geothermal energy", "hydropower", "solar energy", "wind energy", "electric vehicles", "evs", "sustainable agriculture", "circular economy", "climate education", "climate communication", "climate technology", "climate finance", "climate impact assessment", "biodiversity loss", "urban heat island effect", "green building standards", "climate data", "climate action plans"]
transportation_taxonomy = ["transportation", "highways", "road transportation", "traffic management", "traffic safety", "traffic congestion", "road signs and markings", "road maintenance", "bridges", "tunnels", "public transportation", "mass transit", "bus rapid transit", "brt", "subways and metro systems", "light rail transit", "lrt", "commuter rail", "ferry services", "aviation", "airports", "airlines", "air traffic control", "airport security", "rail transportation", "freight rail", "passenger rail", "high-speed rail", "intermodal transportation", "ports", "maritime shipping", "containerization", "inland waterways", "shipping", "logistics", "supply chain management", "last mile delivery", "electric vehicles", "evs", "autonomous vehicles", "avs", "connected vehicles", "smart transportation", "transportation infrastructure financing", "transportation planning", "transportation policy", "traffic engineering", "transportation demand management", "tdm", "intelligent transportation systems", "its", "environmental impact assessment", "eia", "transportation safety regulations", "road pricing", "vehicle emissions", "fuel efficiency", "alternative fuels"]
environment_taxonomy = ["environment", "air quality", "water quality", "soil quality", "environmental impact assessment", "eia", "environmental regulations", "environmental policy", "environmental management systems", "ems", "sustainability", "climate change", "greenhouse gas emissions", "carbon footprint", "renewable energy", "energy efficiency", "waste management", "recycling", "hazardous waste", "pollution control", "clean water act", "clean air act", "endangered species", "biodiversity", "conservation", "natural resource management", "ecosystem services", "ecological restoration", "environmental monitoring", "environmental justice", "environmental education", "sustainable development goals", "sdgs", "circular economy", "green technologies", "low-carbon economy", "carbon sequestration", "ocean conservation", "marine protected areas", "mpas", "land use planning", "urban sustainability", "industrial ecology", "ecotourism", "sustainable agriculture", "environmental certification", "e.g., leed", "corporate social responsibility", "csr", "eco-labeling", "life cycle assessment", "lca", "environmental remediation", "sustainable fisheries", "wetland conservation", "forest management", "air pollution control", "water resource management"]
aviation_taxonomy = ["aviation", "commercial aviation", "general aviation", "airlines", "airports", "airport operations", "air traffic control", "atc", "flight operations", "aircraft", "fixed-wing aircraft", "rotary-wing aircraft", "helicopters", "aircraft manufacturing", "aircraft design", "aircraft maintenance", "aviation safety", "flight safety", "aviation regulations", "international civil aviation organization", "icao", "federal aviation administration", "faa", "european union aviation safety agency", "easa", "aviation security", "airport security", "aviation fuel", "aviation weather", "airline operations", "airline alliances", "airline fleet", "airline routes", "airline revenue management", "passenger experience", "cabin crew", "pilot training", "flight planning", "aircraft leasing", "air cargo", "cargo aircraft", "unmanned aerial vehicles", "uavs", "drone operations", "aviation insurance", "aviation finance", "aviation law", "airline codeshare", "aviation maintenance repair and overhaul", "mro", "aviation noise", "aviation emissions", "aviation sustainability", "aviation accidents", "air traffic management", "atm", "airport infrastructure", "aviation communication", "aviation technology"]
data_analytics_taxonomy = ["data analytics", "descriptive analytics", "predictive analytics", "prescriptive analytics", "diagnostic analytics", "data mining", "data visualization", "big data analytics", "machine learning", "artificial intelligence", "natural language processing", "time series analysis", "regression analysis", "classification", "clustering", "anomaly detection", "data warehousing", "etl", "extract, transform, load", "business intelligence", "data governance", "data quality", "data integration", "data modeling", "data lake", "data mart", "data pipeline", "data cleansing", "data transformation", "data enrichment", "data aggregation", "statistical analysis", "hypothesis testing", "a/b testing", "data sampling", "data imputation", "data segmentation", "feature engineering", "data normalization", "data standardization", "data encryption", "data masking", "data anonymization", "data auditing", "data lineage", "data catalog", "data stewardship", "data retention", "data archiving", "data recovery", "data privacy", "data security"]
generative_ai_taxonomy = ["generative ai", "text generation", "image generation", "music generation", "video generation", "code generation", "natural language processing", "nlp", "transformer models", "generative adversarial networks", "gans", "variational autoencoders", "vaes", "reinforcement learning", "language models", "neural networks", "deep learning", "few-shot learning", "zero-shot learning", "transfer learning", "prompt engineering", "fine-tuning", "self-supervised learning", "unsupervised learning", "semi-supervised learning", "style transfer", "image super-resolution", "text-to-speech", "tts", "speech-to-text", "stt", "multimodal models", "text summarization", "text translation", "question answering", "conversational ai", "sentiment analysis", "emotion recognition", "content generation", "synthetic data", "data augmentation", "ethical ai", "bias mitigation", "model interpretability", "explainable ai", "xai", "human-ai interaction", "creativity in ai", "ai in art", "ai in writing", "ai in gaming", "ai in design", "ai in marketing", "ai in education", "ai in healthcare", "ai for accessibility", "ai ethics and governance"]
cyber_security_taxonomy = ["cyber security", "network security", "information security", "endpoint security", "application security", "cloud security", "data security", "identity and access management", "iam", "threat intelligence", "vulnerability management", "security information and event management", "siem", "intrusion detection systems", "ids", "intrusion prevention systems", "ips", "firewalls", "antivirus and antimalware", "encryption", "data loss prevention", "dlp", "security operations center", "soc", "penetration testing", "incident response", "cyber threat hunting", "security orchestration, automation, and response", "soar", "zero trust architecture", "multi-factor authentication", "mfa", "public key infrastructure", "pki", "secure socket layer", "ssl) / transport layer security", "tls", "cyber risk management", "compliance management", "security policy management", "mobile security", "internet of things", "iot) security", "industrial control systems", "ics) security", "cyber forensics", "red teaming", "blue teaming", "purple teaming", "cybersecurity awareness training", "cyber hygiene", "security auditing", "phishing prevention", "ransomware protection", "social engineering defense", "insider threat management", "cyber resilience", "supply chain security", "endpoint detection and response", "edr", "managed detection and response", "mdr", "behavioral analytics", "advanced persistent threats", "apt) defense", "cybersecurity frameworks", "e.g., nist, iso/iec 27001", "security governance"]
regulatory_policy_taxonomy = ["regulatory policy", "compliance management", "risk assessment", "regulatory compliance", "policy development", "regulatory affairs", "regulatory frameworks", "regulatory auditing", "compliance monitoring", "internal controls", "regulatory reporting", "policy implementation", "governance, risk, and compliance", "grc", "anti-money laundering", "aml", "know your customer", "kyc", "data protection regulations", "environmental regulations", "health and safety regulations", "financial regulations", "consumer protection regulations", "trade compliance", "export control regulations", "import regulations", "industry standards", "ethical standards", "legal compliance", "corporate governance", "compliance training", "compliance culture", "regulatory change management", "regulatory impact analysis", "regulatory liaison", "licensing and permits", "regulatory risk management", "quality assurance", "regulatory strategy", "regulatory technology", "regtech", "transparency regulations", "whistleblower protection", "privacy regulations", "anti-bribery and corruption", "abc", "cybersecurity regulations", "intellectual property regulations", "labor and employment regulations", "tax compliance", "competition law", "compliance analytics", "standards compliance", "sustainable compliance", "regulatory intelligence", "ethical compliance"]
enterprise_cloud_solutions_taxonomy = ["enterprise cloud solutions", "infrastructure as a service", "iaas", "platform as a service", "paas", "software as a service", "saas", "cloud computing", "public cloud", "private cloud", "hybrid cloud", "multi-cloud", "cloud migration", "cloud management", "cloud security", "cloud storage", "cloud networking", "cloud databases", "cloud orchestration", "cloud automation", "cloud scalability", "cloud load balancing", "cloud backup", "disaster recovery as a service", "draas", "cloud cost management", "cloud compliance", "cloud monitoring", "cloud analytics", "cloud devops", "serverless computing", "edge computing", "containerization", "kubernetes", "microservices architecture", "api management", "identity and access management", "iam", "virtual private cloud", "vpc", "elasticity", "high availability", "fault tolerance", "cloud governance", "cloud service level agreements", "slas", "cloud integration", "cloud native applications", "cloud performance optimization", "content delivery network", "cdn", "virtual machines", "vms", "cloud app development", "cloud security posture management", "cspm", "unified endpoint management", "uem", "cloud collaboration tools", "managed cloud services", "cloud data encryption", "cloud innovation"]
research_evaluation_taxonomy = ["research evaluation", "peer review", "bibliometrics", "scientometrics", "altmetrics", "citation analysis", "impact factor", "h-index", "research impact", "research quality assessment", "research productivity", "funding evaluation", "grant reviews", "publication analysis", "research metrics", "research output", "collaboration networks", "co-authorship analysis", "research influence", "research utilization", "research translation", "research dissemination", "knowledge transfer", "knowledge mobilization", "research uptake", "research ethics", "research governance", "research integrity", "research performance indicators", "research policy", "research strategy", "evaluation frameworks", "program evaluation", "outcome evaluation", "process evaluation", "formative evaluation", "summative evaluation", "qualitative evaluation", "quantitative evaluation", "mixed-methods evaluation", "logic models", "theory of change", "evaluation capacity building", "evaluation standards", "evaluation utilization", "stakeholder engagement", "evaluation reporting", "cost-benefit analysis", "cost-effectiveness analysis", "evaluation ethics", "evidence-based policy"]
advisory_taxonomy = ["advisory", "management consulting", "financial advisory", "strategic planning", "business transformation", "risk management", "compliance advisory", "it consulting", "human resources consulting", "marketing consulting", "operations consulting", "supply chain management", "change management", "leadership development", "organizational development", "performance improvement", "merger and acquisition advisory", "due diligence", "restructuring advisory", "tax advisory", "wealth management", "investment advisory", "legal advisory", "environmental, social, and governance", "esg) advisory", "digital transformation", "innovation management", "corporate governance", "market entry strategy", "brand strategy", "customer experience", "cx) consulting", "cybersecurity advisory", "data analytics advisory", "economic advisory", "financial planning and analysis", "fp&a", "project management", "public relations advisory", "crisis management", "talent management", "technology strategy", "product development advisory", "sustainability advisory", "business continuity planning", "competitive analysis", "pricing strategy", "regulatory advisory", "audit and assurance services", "knowledge management", "procurement advisory", "training and development", "market research", "strategic sourcing"]
public_health_taxonomy = ["public health", "epidemiology", "biostatistics", "health promotion", "disease prevention", "environmental health", "global health", "health policy", "health systems", "community health", "occupational health", "maternal and child health", "infectious disease control", "chronic disease management", "health education", "behavioral health", "social determinants of health", "health disparities", "public health surveillance", "vaccination programs", "health communication", "nutrition and dietetics", "mental health", "health equity", "emergency preparedness", "health economics", "health services research", "health informatics", "public health ethics", "health literacy", "public health law", "health advocacy", "sanitation and hygiene", "substance abuse prevention", "sexual and reproductive health", "injury prevention", "adolescent health", "aging and geriatrics", "public health genomics", "water quality", "air quality", "vector control", "food safety", "public health nursing", "public health administration", "risk assessment", "community outreach", "health impact assessment", "global health security", "public health nutrition", "program evaluation"]

"""## Flair Code Block"""

def for_doccano_pre_tagging(single_line_paragraph):
  sentence = Sentence(single_line_paragraph)
  tagger.predict(sentence)
  entities = []
  for entity in sentence.get_spans('ner'):
    entities.append([entity.start_position, entity.end_position, entity.labels[0].value])

  results = programatic_taxonomy_detection(single_line_paragraph, startup_taxonomy)
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

  # results = programatic_taxonomy_detection(single_line_paragraph, startup_taxonomy)

  # for result in results:
  #   entity_text_temp = result['Text']
  #   entity_type_temp = result['Type']

  #   entity_text_temp, entity_type_temp = standardize_entity_text(entity_text_temp, entity_type_temp)

  #   entity_text_with_type = entity_text_temp + " (" + entity_type_temp + ")"
  #   entities.append(entity_text_with_type)

  #   if entity_text_temp not in entity_text_only:
  #     entity_text_only.append(entity_text_temp)

  #   if entity_type_temp not in entity_type_only:
  #     entity_type_only.append(entity_type_temp)

  #   if entity_type_temp in cannonical_map:
  #     if entity_text_temp not in cannonical_map[entity_type_temp]:
  #       cannonical_map[entity_type_temp].append(entity_text_temp)
  #   else:
  #     cannonical_map[entity_type_temp] = [entity_text_temp]

  for taxonomy in custom_taxonomies:
    results = programatic_taxonomy_detection(single_line_paragraph, globals()[taxonomy])
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