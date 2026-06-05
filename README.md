# Curling AI

Detta projekt är en förenklad version av curling med en AI-agent som tränas för att spela gula stenar mot en mänsklig spelare som spelar röda stenar. Spelet består av en omgång där varje lag har fyra stenar. Ett skott definieras av tre parametrar: hastighet, vinkel och curl.

AI-modellen kan beskrivas som en variant av en utility-based agent. Den tar emot ett state, överväger en eller flera möjliga actions i form av kandidatskott, simulerar hur dessa actions påverkar spelet och beräknar utility för resultaten. Därefter väljs den action som ger högst förväntad utility. När skottet spelas uppdateras miljön och ett nytt state skickas vidare till nästa beslutspunkt.

## Kör programmet

Projektet är skrivet i Processing.

1. Öppna Processing.
2. Öppna filen `curling/curling.pde`.
3. Starta sketchen med Run.

Bildfilerna för stenarna ligger i `curling/data`. Tränade modeller sparas och laddas från `curling/data/models`.

## Spel

Den mänskliga spelaren spelar rött. Spelaren väljer curl, vinkel och fart och låser sedan skottet med mätaren. Högre curl gör mätaren snabbare och svårare att träffa rätt.

AI:n spelar gula stenar. I UI:t går det att välja mellan två AI-lägen:

- `MC AI`: testar många slumpmässiga skott, simulerar dem och väljer det skott som ser bäst ut direkt efter simuleringen.
- `NN`: använder den tränade neurala modellen med expertnätverk, selector-nätverk och långsiktig utility.

Knapparna `Spara` och `Ladda` används för att spara eller ladda tränade modeller. Modeller sparas som `.curlmodel`-filer i `data/models`.

## AI-modellen

State representerar det aktuella spelläget. Varje sten konverteras till numeriska värden som beskriver stenens position, vilket lag stenen tillhör och om stenen finns i spel. Dessutom läggs information till om hur många stenar som är kvar och vilket lag som spelade senaste stenen. Dessa värden normaliseras så att de blir lättare för nätverket att hantera.

AI-modellen använder flera specialiserade expertnätverk. Varje expert är tränad mot en viss typ av skott, till exempel draw, takeout, guard eller freeze.

För att välja vilka experter som är relevanta används ett selector-nätverk. Selector-nätverket tar emot samma state-representation som experterna, men producerar inte ett skott. Istället producerar det en sannolikhetsfördelning över experterna med hjälp av softmax. Under spel används de mest sannolika experterna som kandidater.

## Träning

Träningen är en förenklad, policy-gradient-inspirerad metod. Nätverket förutspår först ett skott för ett givet spelläge. Därefter genereras flera små variationer av detta skott genom att justera curl, hastighet och vinkel slumpmässigt. Alla variationer simuleras i fysikmotorn och poängsätts med heuristiker.

De bästa variationerna väljs ut som elite shots. Därefter beräknas skillnaden mellan modellens ursprungliga output och medelvärdet av de bästa skotten. Denna skillnad används som gradient för att justera nätverkets vikter genom backpropagation.

Träningen använder curriculum learning. Det betyder att modellen först tränas på enklare lägen nära slutet av omgången och sedan gradvis tränas på tidigare och mer komplexa lägen. På så sätt lär sig modellen först enklare slutlägen innan den tränas på mer strategiska situationer.

## Långsiktig utility

AI-modellen använder principen för maximum expected utility för att värdera långsiktiga konsekvenser.

När AI:n värderar ett skott simuleras först det aktuella skottet. Om omgången inte är slut analyseras sedan motståndarens mest sannolika svarsskott med hjälp av selector-nätverket. De mest sannolika svaren viktas efter sina sannolikheter. Från dessa svar simuleras resten av omgången hela vägen till slutet med de neurala nätverken.

För att undvika för många grenar i simuleringen förgrenas bara nästa motståndarskott. Därefter simuleras resten av omgången utan ytterligare förgreningar genom att selector-nätverket väljer den mest sannolika experten vid varje skott.

Samma långsiktiga värdering används även i träningen genom `FinalScoreHeuristic`. Den kombineras med varje experts egen heuristik. Expertens egen heuristik belönar skotttypens mål, medan `FinalScoreHeuristic` uppskattar hur bra skottet förväntas bli när resten av omgången spelats färdigt.

## Projektstruktur

- `curling/curling.pde`: huvudfil, setup, draw-loop, spellogik och AI-val.
- `curling/NeuralPolicy.pde`: neuralt nätverk som producerar skott.
- `curling/GradientEnsemble.pde`: expertnätverk och long-term utility.
- `curling/ShotTypeSelector.pde`: selector-nätverk som väljer sannolikheter över experter.
- `curling/PolicyGradientTraining.pde`: träning av expertnätverk.
- `curling/Heuristic.pde`: heuristiker för draw, takeout, guard, freeze och final score.
- `curling/Physics.pde`: fysiksimulering för stenar och kollisioner.
- `curling/ModelStorage.pde`: spara och ladda modeller.
