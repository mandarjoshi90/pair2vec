from torchtext.data import Field, Iterator, TabularDataset


def read_data(config):
    args = Field(lower=True, tokenize='spacy') if config.compositional_args else Field()
    rels = Field(lower=True, tokenize='spacy') if config.relational_args else Field()
    
    #TODO we will need to add a header to the files
    data = TabularDataset(path=config.data_path, format='tsv', fields = [('subject', args), ('relation', rels), ('object', args)])
    train, dev = data.split(split_ratio=0.99)
    print('Train size:', len(train), '   Dev size:', len(dev))
    
    args.build_vocab(train)
    rels.build_vocab(train)
    config.n_args = len(args.vocab)
    config.n_rels = len(rels.vocab)
    print("#Args:", config.n_args, "   #Rels:", config.n_rels)
    
    train_iter, dev_iter = Iterator.splits((train, dev), batch_size=config.batch_size, device=args.gpu)
    train_iter.repeat = False
    
    #TODO need to figure out how to duplicate the relations field, and then detach it from the regular order. This'll allow us to effectively sample relations.
    
    return train_iter, dev_iter
