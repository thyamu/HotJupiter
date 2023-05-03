header = ['Metallicity', 'Altitude',
        'Mean Degree',
        'CO Degree', 'CH4 Degree', 'NH3 Degree', 'H2O Degree',
        'Average shortest path length',
        'Average clustering coefficient',
        'CO clustering coefficient', 'CH4 clustering coefficient', 'NH3 clustering coefficient','H2O clustering coefficient',
        'CO node betweenness centrality', 'CH4 node betweenness centrality', 'NH3 node betweenness centrality',
        'H2O node betweenness centrality',
        'Edge betweenness centrality',
        'Average neighbor degree',
        'CO neighbor degree', 'CH4 neighbor degree', 'NH3 neighbor degree', 'H2O neighbor degree',
        'CO abundance', 'CH4 abundance', 'NH3 abundance', 'H2O abundance',
        'Delta G distribution', 'Phi distribution',
        'Average node betweenness centrality', 'Temperature', 'kzz']

header_average = [
    'Mean Degree', 'Average shortest path length', 'Average clustering coefficient',
    'Average neighbor degree','Average node betweenness centrality', 'Edge betweenness centrality']

header_abundance = [n for n in header if n.find('abundance') > -1] # 'CO abundance', 'CH4 abundance', 'NH3 abundance', 'H2O abundance']

header_CO = [n for n in header if n.find('CO') > -1]
header_CO_without_abundance = list(header_CO)
header_CO_without_abundance.remove('CO abundance')

header_CH4 = [n for n in header if n.find('CH4') > -1]
header_CH4_without_abundance = list(header_CH4)
header_CH4_without_abundance.remove('CH4 abundance')

header_NH3 = [n for n in header if n.find('NH3') > -1]
header_NH3_without_abundance = list(header_NH3)
header_NH3_without_abundance.remove('NH3 abundance')

header_H20 = [n for n in header if n.find('H2O') > -1]
header_H20_without_abundance = list(header_H20)
header_H20_without_abundance.remove('H2O abundance')

header_individual_cc = [n for n in header if n.find('clustering coefficient') > -1]
header_individual_cc.remove('Average clustering coefficient')

header_individual_betweenness = [n for n in header if n.find('node betweenness centrality') > -1]
header_individual_betweenness.remove('Average node betweenness centrality')

header_individual_degree = [n for n in header if n.find('Degree') > -1 and n.find('neighbor degree') == -1]
header_individual_degree.remove('Mean Degree')

header_individual_neighborDegree = [n for n in header if n.find('neighbor degree') > -1]
header_individual_neighborDegree.remove('Average neighbor degree')