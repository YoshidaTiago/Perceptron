# Perceptron

[perceptron_simples](docs/Perceptron_Simples.pdf) - Montado sem bibliotecas, resolve a função lógica AND.

perceptron_pronto - Montado utilizando a biblioteca sklearn, também resolve a função lógica AND.

[perceptron_flowers](docs/Classificacao_de_Flores_com_Perceptron.pdf) - Montado sem bibliotecas, classifica as flores com a base de dados do Saulo.

perceptron_pronto2 - Montado utilizando a biblioteca sklearn, classifica as flores com a base de dados do Saulo.

## Comparação entre Implementação Manual e com sklearn

### Função lógica AND

A implementação manual se mostrou mais eficiente, pois conseguiu otimizar os pesos e o bias em apenas 3 iterações, enquanto o modelo implementado com a biblioteca sklearn levou 8 iterações para convergir para a solução correta.

### Classificação de Flores

O modelo manual foi menos eficaz. Ele apresentou grande dificuldade em classificar corretamente amostras da espécie Versicolor e um desempenho razoável com a espécie Virginica. Já o programa montado com sklearn, teve somente dificuldade com flores versicolor.