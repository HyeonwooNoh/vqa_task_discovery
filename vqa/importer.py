def get_model_types():
    model_types = [
        'proposed',
        'wordnet',
        'description',
        'separable',
        'answer-embedding',
        'standard-vqa',
        'proposed-seen-in-test',
    ]
    return model_types


def get_model_class(model_type='vqa'):
    if model_type == 'proposed':
        from vqa.model_proposed import Model
    elif model_type == 'wordnet':
        from vqa.model_proposed import Model
    elif model_type == 'description':
        from vqa.model_proposed import Model
    elif model_type == 'separable':
        from vqa.model_separable import Model
    elif model_type == 'answer-embedding':
        from vqa.model_answer_embedding import Model
    elif model_type == 'standard-vqa':
        from vqa.model_standard import Model
    elif model_type == 'proposed-seen-in-test':
        from vqa.model_proposed_seen_in_test import Model
    else:
        raise ValueError('Unknown model_type: {}'.format(model_type))
    return Model
