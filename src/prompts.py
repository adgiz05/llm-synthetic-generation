PROMPTS = {
    #---------------------------------------------------------------------CMSB----------------------------------------------------------------------------------------------#
    
    'cmsb_p2p' : """You are an expert sociologist in sexism content detection. This sentence: "{text}" is considered sexist. Your goal is to propose new sexist sentences preserving the writing style and the informal slang. You should provide {n_samples} different and diverse options. Try not to repeat same hashtags and/or names. Make sure they are sexist. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputing the sentences without explaining why you create them.""",
    
    'cmsb_n2n' : """You are an expert in generate negatives samples for a classification dataset. Take this sentence: "{text}" as an example for the generation. Your goal is to propose {n_samples} new samples. The samples should talk about another topic than the original preserving the informal slang, the expressions and the writing style. Also the samples have to maintain the original discriminatory tone. They should be different and diverse with each other. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'cmsb_p2n' : """You are an expert sociologist in sexism content detection. This sentence: "{text}" is considered sexist. Your goal is to subtract any sexism patterns from the original sentence proposing new sentences. The generated sentences must talk about a different topic. You should provide {n_samples} different and diverse options. Try not to repeat same hashtags and/or names. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'cmsb_n2p' : """You are an expert sociologist in sexism content detection. This sentence: "{text}" is clearly non-sexist, however, your goal is to use the same writing style to generate extreme sexist sentences. You should provide {n_samples} different and diverse options. Try not to repeat same hashtags and/or names. Present the sentences in a numbered list format (Each line like 1. <sentence>).""",
    
    #-----------------------------------------------------------------ETHOS_BINARY------------------------------------------------------------------------------------------#
    
    'ethos_binary_p2p' : """You are an expert in hate speech content detection. This sentence: "{text}" is considered hate speech. Your goal is to propose new hate speech sentences preserving the writing style and the informal slang. You should provide {n_samples} different and diverse options. Make sure that they promote hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputing the sentences without explaining why you create them.""",
    
    'ethos_binary_n2n' : """You are an expert in generate negatives samples for a classification dataset. Take this sentence: "{text}" as an example for the generation. Your goal is to propose {n_samples} new samples. The samples should talk about another topic than the original preserving the informal slang, the expressions and the writing style. Also the samples have to maintain the original tone. They should be different and diverse with each other. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'ethos_binary_p2n' : """You are an expert in hate speech content detection. This sentence: "{text}" is considered hate speech. Your goal is to subtract any hateful patterns from the original sentence proposing new sentences. The generated sentences must talk about a different topic preserving the informal slang, the expressions and the writing style. You should provide {n_samples} different and diverse options. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'ethos_binary_n2p' : """You are an expert in hate speech content detection. This sentence: "{text}" is clearly not prompoting hate speech, however, your goal is to use the same writing style to generate extreme hate speech sentences. You should provide {n_samples} different and diverse options. Make sure that they promote hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>).""",
    
    
    #-----------------------------------------------------------------ANTIASIAN------------------------------------------------------------------------------------------#
    
    'antiasian_p2p' : """You are an expert in antiasian hate speech content detection. This sentence: "{text}" is considered antiasian hate speech. Your goal is to propose new antiasian hate speech sentences preserving the writing style and the informal slang. You should provide {n_samples} different and diverse options. Make sure that they promote antiasian hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputing the sentences without explaining why you create them.""",
    
    'antiasian_n2n' : """You are an expert in generate negatives samples for a classification dataset. Take this sentence: "{text}" as an example for the generation. Your goal is to propose {n_samples} new samples. The samples should talk about another topic than the original preserving the informal slang, the expressions and the writing style. Also the samples have to maintain the original tone. They should be different and diverse with each other. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'antiasian_p2n' : """You are an expert in antiasian hate speech content detection. This sentence: "{text}" is considered antiasian hate speech. Your goal is to subtract any hateful patterns from the original sentence proposing new sentences. The generated sentences must talk about a different topic preserving the informal slang, the expressions and the writing style. You should provide {n_samples} different and diverse options. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'antiasian_n2p' : """You are an expert in antiasian hate speech content detection. This sentence: "{text}" is clearly not prompoting antiasian hate speech, however, your goal is to use the same writing style to generate extreme antiasian hate speech sentences. You should provide {n_samples} different and diverse options. Make sure that they promote antiasian hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>).""",
    
    #-----------------------------------------------------------------STORMFRONT------------------------------------------------------------------------------------------#
    
    'stormfront_p2p' : """You are an expert in white supremacy hate speech content detection. This sentence: "{text}" is considered white supremacy hate speech. Your goal is to propose new white supremacy hate speech sentences preserving the writing style and the informal slang. You should provide {n_samples} different and diverse options. Make sure that they promote white supremacy hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputing the sentences without explaining why you create them.""",
    
    'stormfront_n2n' : """You are an expert in generate negatives samples for a classification dataset. Take this sentence: "{text}" as an example for the generation. Your goal is to propose {n_samples} new samples. The samples should talk about another topic than the original preserving the informal slang, the expressions and the writing style. Also the samples have to maintain the original tone. They should be different and diverse with each other. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'stormfront_p2n' : """You are an expert in white supremacy hate speech content detection. This sentence: "{text}" is considered white supremacy hate speech. Your goal is to subtract any hateful patterns from the original sentence proposing new sentences. The generated sentences must talk about a different topic preserving the informal slang, the expressions and the writing style. You should provide {n_samples} different and diverse options. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputting the sentences without explaining why you create them.""",
    
    'stormfront_n2p' : """You are an expert in white supremacy hate speech content detection. This sentence: "{text}" is clearly not prompoting white supremacy hate speech, however, your goal is to use the same writing style to generate extreme antiasian hate speech sentences. You should provide {n_samples} different and diverse options. Make sure that they promote white supremacy hate speech. Present the sentences in a numbered list format (Each line like 1. <sentence>).""",
    
    #----------------------------------------------------------------LABELING-------------------------------------------------------------------------------------------#

    'cmsb_label': """Your goal is to label the following samples as SEXIST or NEUTRAL. If a sample contains one or more of the following features, the sample is considered sexist, otherwise is just neutral. You should provide a short reasoning about why you label the sample as SEXIST or NEUTRAL. The format should be in two lines: in the first one you provide the reasoning and in the second one just the label (SEXIST or NEUTRAL). An example of the format could be:
    REASON: The sample contains stereotypes and comparisons between...
    LABEL: SEXIST
    
    or
    
    REASON: The sample does not contain any sexist patterns...
    LABEL: NEUTRAL
    
    You should provide the labelling for the following sample: {text}""",

    #-----------------------------------------------------------------AUGGPT------------------------------------------------------------------------------------------#

    'auggpt_single_turn': "Please rephrase the following sentence: {text}. ",
}

