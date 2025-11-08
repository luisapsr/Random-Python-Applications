from transformers import pipeline

print("✅ Transformers funcionando!")

# cria um pipeline de análise de sentimento (modelo padrão em inglês)
analisador = pipeline("sentiment-analysis")

# roda um teste simples
resultado = analisador("Eu estou muito feliz hoje!")
print(resultado)