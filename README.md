# IA_Aulas
Repositório de aprendizado de máquina em Python.

## Pré-requisitos
- Python 3.11.9
- `pip`

## Como usar
1. Criar o ambiente virtual:

```powershell
python -m venv .venv
```
2. Ativar o ambiente virtual:
```powershell
.venv\Scripts\activate.ps1
```
3. Instalar as dependências:
```powershell
pip install -r "requirements.txt"
```

## Como executar os exemplos
Substitua o caminho pelo arquivo que você quer testar.
O caminho que leva ao dataset pode, e provavelmente será, diferente.

## add
As aulas vistas como base para todo conteudo desse repositorio normalmente colocavam a parte de reajuste de dados antes da sua divisão para o treinamento. Isso pode fazer com que haja um "leak" de dados de teste na hora do treinamento. Dito isso, quase todo arquivo dos basicas e engenharia de atributos terá uma versão que é como a aula de referencia e um que é feito com um ajuste para não haver esse erro que as vez3es causam diferenças no resultado real das metricas de classificação
