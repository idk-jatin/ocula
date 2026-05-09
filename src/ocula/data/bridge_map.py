LABEL2ID ={
    "hate" : 0,
    "offensive" : 1,
    "normal" :2
}

ID2LABEL = {v:k for k,v in LABEL2ID.items()}

NUM_LABELs = 3

BRIDGE = {
"hatexplain":{
    "hatespeech":"hate",
    "offensive":"offensive",
    "normal":"normal"
},
"davidson":{
    "0":"hate",
    "1":"offensive",
    "2":"normal"
},
    "hasoc_hindi": {
        "HATE": "hate",
        "OFFN": "offensive",
        "PRFN": "offensive",
        "NONE": "normal",
    },

    "hasoc_english": {
        "HATE": "hate",
        "OFFN": "offensive",
        "PRFN": "offensive",
        "NONE": "normal",
    },

    "indo_hate": {
        "'HS0'": "normal",
        "'HS1'": "offensive",
        "'HSN'": "hate",
        "HS0": "normal",
        "HS1": "offensive",
        "HSN": "hate",
    },
}

# Function to map the dataset labels to a predifined common labels for all
def map_label(dataset,raw_label):
    bridge = BRIDGE[dataset]
    key = str(raw_label).strip()
    if key not in bridge:
        raise KeyError(
            f"Unknown label '{raw_label}' for dataset '{dataset}'"
            f"Valid: {list(bridge.keys())}"
        )
    return LABEL2ID[bridge[key]]