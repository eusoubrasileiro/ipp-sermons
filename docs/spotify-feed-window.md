> ⚠️ **PESQUISA / EXPLORAÇÃO** — apuração técnica para conversa com a equipe de
> TI da IP Peregrinos. Descreve um comportamento de sistemas de terceiros, não
> uma especificação deste projeto.

# Por que os sermões antigos somem do Spotify

Resumo para quem cuida do site e das publicações da igreja. O problema **não é
do Spotify** e não é da busca — é do feed, e a correção está com a igreja.

## O que está acontecendo

Todas as plataformas onde a igreja publica — Spotify, Apple Podcasts, Deezer e
YouTube Music — leem **um único endereço**, o feed RSS gerado pelo SoundCloud:

```
https://feeds.soundcloud.com/users/soundcloud:users:695742830/sounds.rss
```

Esse feed é **limitado a 500 episódios**. Hoje o mais antigo que ele ainda
carrega é de **06/05/2021**.

Quando um sermão sai dessa janela de 500, os agregadores entendem que o
episódio foi removido e **tiram o episódio do ar**. O endereço continua
existindo, mas responde "não encontrado".

A janela é **rolante**: a cada novo sermão publicado, o mais antigo cai fora.
Como a igreja publica algo entre 50 e 100 gravações por ano, isso significa que
**o acervo antigo desaparece sozinho, aos poucos, sem ninguém ser avisado**.

## Como confirmamos

Testamos os episódios um a um contra o Spotify e comparamos com o feed:

| Situação | Resultado |
|---|---|
| Sermão ainda dentro do feed | **funciona** (HTTP 200) |
| Sermão que saiu da janela de 500 | **fora do ar** (HTTP 404) |
| Sermão de 2021 ainda no feed | **funciona** — mesmo sendo antigo |

A correspondência foi exata em todos os casos testados. Ou seja: **não é a idade
do sermão que decide, é estar ou não dentro do feed.**

Hoje isso atinge **115 dos 543 sermões** que têm episódio no Spotify — todos de
2019, 2020 e início de 2021. E o número cresce sozinho a cada publicação nova.
(A contagem do dia está em `data/facets/spotify_episodes.csv`.)

*Observação de método: o Spotify bloqueia consultas em massa (HTTP 429). Os
números acima vêm de amostras espaçadas, não de uma varredura completa.*

## O que dá para fazer (do lado da igreja)

Em ordem de esforço:

1. **Aumentar ou remover o limite do feed.** Se o plano do SoundCloud permitir
   configurar quantos itens o RSS publica, esse é o conserto direto — os
   episódios antigos voltam a ser distribuídos.
2. **Publicar por um serviço de podcast de verdade** (Spotify for Creators,
   Anchor, Buzzsprout e similares), que mantêm o catálogo inteiro no feed. O
   SoundCloud continuaria como está.
3. **Não fazer nada.** O áudio **não se perde**: o SoundCloud cobre 100% do
   acervo e nada foi apagado. O que se perde é o alcance nas outras
   plataformas.

## O que já fizemos do nosso lado

O site de busca (`ipp-sermons.amiticia.cc`) agora confere o feed e só mostra o
botão do Spotify quando o episódio realmente responde. Antes havia uma regra
por data — "esconder tudo antes de 2022" — que era um chute: escondia 52
episódios de 2021 que funcionavam perfeitamente, e mais cedo ou mais tarde
mostraria episódios já fora do ar.

O botão do SoundCloud aparece sempre, para todos os sermões.
