


def read_data(config):
    inputs = data.Field(lower=config.lower, tokenize='spacy')
    answers = data.Field(sequential=False, unk_token=None)
    
    
    # Load data.
    train, dev, test = datasets.SNLI.splits(inputs, answers)
    print ("Dev data size", len(dev), len(hard_dev))
    
    inputs.build_vocab(train, dev, test)
    if config.word_vectors:
        if os.path.isfile(config.vector_cache):
            inputs.vocab.vectors = torch.load(config.vector_cache)
        else:
            inputs.vocab.load_vectors(config.word_vectors)
            makedirs(os.path.dirname(config.vector_cache))
            torch.save(inputs.vocab.vectors, config.vector_cache)
    answers.build_vocab(train)
    print (type(answers), answers.vocab.stoi, answers.vocab.itos)
    
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=config.batch_size, device=args.gpu)
    print (dev_iter)
    train_iter.repeat = False
    
    
    
    if config.compositional_args:
    
    else:
    
    
    if config.compositional_rels:
    
    else:
    
    
    # Some arguments.
    config.n_args = len(args.vocab)
    config.n_rels = len(rels.vocab)
    print("#Args:", config.n_args, "   #Rels:", config.n_rels)
    
    return train_iter, dev_iter