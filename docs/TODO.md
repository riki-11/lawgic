# TODO

### Testing the Finetuned Models
- (For topic classification) Literally just copy-paste a sentence from a ToS from a service that isn't on the dataset. Then just check it for myself... That's it.
    - I can test on AI services! Like OpenAI
    - OR USE THESE: https://zenodo.org/records/15013541 and https://zenodo.org/records/15014823?preview_file=training_tosdr_privacy_data.csv
- (For harm score) then ask it it to give a rating as well... then compare it with what I would personally give as a rating.
    - It seems like LEGAL-BERT's harm scoring might not be the best...? So I might still need to use a decoder model to be a double check... but the classifier seems alright.
    - (Well, it's not ALL bad... it it jsut not 100% accurate, which is to be expected I guess.)

## API 
    - Maybe can turn transformer inference into lawgic package soon.