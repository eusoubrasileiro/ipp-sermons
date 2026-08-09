> ⚠️ **RASCUNHO NÃO RATIFICADO** — desenho pronto, execução adiada por decisão
> do dono. Não é especificação: nada aqui foi aprovado para ser construído.

# Qualidade da classificação — o que ainda não medimos

Duas perguntas abertas sobre as passadas de LLM, e a metade do instrumento que
falta para respondê-las. Escrito porque o trabalho está pronto para ser feito e
não há pressa para fazê-lo — não porque ficou pela metade.

## O que já existe

- **Contabilidade de tokens** (`backend/src/lib/usage.ts`): todo script que fala
  com a OpenRouter imprime tokens e dólares reais no fim.
- **Bancada de comparação** (`pnpm compare:topics`): reclassifica uma amostra de
  sermões já classificados sob várias configurações e diz onde elas divergem.
  Não grava nada no corpus.

## O que falta, e por quê

`pnpm eval` prova que a **busca** continua boa — recall@10 sobre um conjunto de
consultas de gabarito. Não existe nada equivalente para a **classificação**. Se
o modelo começar a errar a passagem bíblica ou a escolher temas frouxos, nenhum
gate percebe: o CSV é bem formado, a carga funciona, as páginas renderizam.

Isso importa mais desde que `pnpm corpus:update` roda sem paradas.

---

## Passo 1 — julgar as divergências (≈20 min, uma vez)

```bash
pnpm compare:topics --sample 40    # ~$0,11, escreve .compare/topics-<data>-n<amostra>.csv
```

O CSV já vem filtrado: só as linhas em que as configurações divergem —
tipicamente um terço da amostra. Abra numa planilha, leia os temas propostos em
cada coluna `<config>_temas` e escreva o id da configuração vencedora (`A` a
`E`) na coluna `melhor`. Deixe em branco quando não houver diferença que
importe.

Contar os votos ainda é manual: não existe `--score`. É trabalho do Passo 2, e
é pequeno.

**As duas perguntas que esse julgamento responde:**

1. **Transcript inteiro vale mais que a amostra de 3 janelas?** A amostra guarda
   51% do texto e economiza $0,26 no corpus inteiro — não por rodada. Se o texto
   inteiro vencer, a economia não se justifica.
2. **Um modelo mais barato serve?** Os candidatos com `structured_outputs` real
   custam entre um décimo e um terço do `gpt-5.6-luna`. Trocar sem medir é
   apostar ground truth commitado.

Reclassificar o corpus inteiro custa ~$1 e continua sendo decisão do dono,
depois disso.

## Passo 2 — o julgamento vira gate (≈1h30 de código)

As linhas julgadas são verdade conhecida. Congelá-las num gabarito transforma
uma opinião numa regressão detectável.

- `backend/test/golden/facets.json` — mesmo espírito do `queries.json`: sermão,
  temas esperados, e o porquê em uma linha.
- `backend/src/scripts/eval-facets.ts` — espelha `eval-golden.ts`: reclassifica
  os sermões do gabarito com a configuração atual, reporta, `exit 1` se divergir.
- `scripts/corpus-update.sh` — o estágio `eval` passa a rodar os dois.

`backend/test/golden/**` é caminho crítico com `Ratified-by:`, o que é a proteção
certa: editar o gabarito para uma regressão passar é exatamente a falha que ele
existe para pegar.

Verificação: verde com a configuração atual, e **depois adulterar uma linha do
gabarito de propósito para confirmar que fica vermelho** — um gate que nunca
falhou não está provado.

---

## Contexto que não deve se perder

**A mistura é o risco, não o custo.** Classificar só os sermões novos com outro
modelo produz a mesma doença que `tools/corpus-update/CLAUDE.md` proíbe para as
métricas do corpus: *"different tooling silently makes new rows incomparable to
the existing corpus"*. Se um modelo for mais generoso que o outro,
`/temas/ansiedade` passa a super-representar sermões recentes e nada acusa. O
corpus é pequeno o bastante para que reclassificar tudo seja mais barato que
conviver com a inconsistência.

**`structured_outputs` não é `response_format`.** Só o primeiro garante
`strict: true`, e é ele que faz o enum fechado impedir o modelo de inventar um
livro da Bíblia ou um tema fora da taxonomia. Vários modelos baratos anunciam
apenas o segundo — `qwen/qwen3.7-flash` entre eles. Confira o campo antes de
pôr um candidato na bancada.

**O aviso contra o DeepSeek em `llm.ts` cita issues do V3.** O V4 Flash é outra
geração e entra na bancada como candidato. Se for absolvido, o comentário muda
junto; até lá, o aviso vale.
