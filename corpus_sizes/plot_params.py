# Set all variables to test:
stats = ['Dataset',
         'Word count',
         'Window',
         'Dimension',
         'Sampling']


# Set x-axis and bars for graphs
graph_dimensions = [
    #('Word count', 'Bins'),
    ('Word count', 'Window'),
    #('Word count', 'Sampling'),
    #('Word count', 'Cross-sentence'),
    #('Window', 'Sampling'),
    #('Window', 'Cross-sentence'),
    #('Window', 'Sampling')
]

dimension_values_map = {
    'Bins': [
        "Low bin score",
        "Middle bin score",
        "High bin score",
        "Mixed bin score",
        "General score"
    ],
    'Window': [2, 5, 10],
    'Sampling': ['hs', 'ns'],
    'Cross-sentence': ['Yes', 'No']

}