# Perceptron

[perceptron_simples](docs/Perceptron_Simples.pdf) - Montado sem bibliotecas, resolve a função lógica AND.

perceptron_pronto - Montado utilizando a biblioteca sklearn, também resolve a função lógica AND.

[perceptron_flowers](docs/Classificao_de_Flores_com_Perceptron.pdf) - Montado sem bibliotecas, classifica as flores com a base de dados do Saulo.

perceptron_pronto2 - Montado utilizando a biblioteca sklearn, classifica as flores com a base de dados do Saulo.

## Comparação entre manual e feitos com bibliotecas

### Função lógica AND

Manual se mostrou mais eficiente, pois demora 3 iterações para otimizar os pesos e bias, enquanto o programa construído com a biblioteca sklearn demora 8.

### Classificação de Flores

Manual se mostrou menos efetivo, tendo alta dificuldade em classificar flores versicolor e baixa dificuldade com flores virginica. Já o programa montado com sklearn, teve somente dificuldade com flores versicolor.