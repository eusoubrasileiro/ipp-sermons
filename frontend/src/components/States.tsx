import { Card } from "./Card.tsx";
import { AlertIcon, EmptyIcon } from "./Icons.tsx";

/** Loading, empty, error and first-visit states. Every one of them is designed, none is blank. */

/** Mirrors the real card's shape so results do not shift the page when they land. */
export function ResultsSkeleton() {
  return (
    <div className="space-y-3" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <Card key={i}>
          <div className="h-5 w-3/4 animate-pulse rounded bg-muted" />
          <div className="mt-2 h-4 w-1/2 animate-pulse rounded bg-muted" />
          <div className="mt-4 space-y-2">
            <div className="h-3.5 w-full animate-pulse rounded bg-muted" />
            <div className="h-3.5 w-full animate-pulse rounded bg-muted" />
            <div className="h-3.5 w-5/6 animate-pulse rounded bg-muted" />
          </div>
          <div className="mt-5 flex gap-2">
            <div className="h-11 w-36 animate-pulse rounded-md bg-muted" />
            <div className="h-11 w-36 animate-pulse rounded-md bg-muted" />
          </div>
        </Card>
      ))}
    </div>
  );
}

export function EmptyState({
  query,
  acao,
}: {
  query: string;
  /**
   * The way out, when there is one. A search narrowed by chips usually returns
   * nothing because of the chips, and telling someone to "try broader words"
   * when the real cause is a filter they set is a dead end.
   */
  acao?: { label: string; onClick: () => void } | undefined;
}) {
  return (
    <Card variant="dashed" className="text-center">
      <EmptyIcon className="mx-auto h-8 w-8 text-muted-foreground" />
      <p className="mt-3 font-medium">Nenhum sermão encontrado para “{query}”.</p>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        {acao
          ? "Os filtros podem estar estreitos demais. Remova um deles acima, ou comece de novo."
          : "Tente palavras mais gerais (“perdão” em vez de “perdoar o irmão”), o nome de um livro da Bíblia, ou o nome do pregador."}
      </p>
      {acao ? (
        <button
          type="button"
          onClick={acao.onClick}
          className="mt-4 min-h-11 rounded-md border border-border bg-card px-4 text-sm font-medium transition hover:bg-accent hover:text-accent-foreground"
        >
          {acao.label}
        </button>
      ) : null}
    </Card>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-col gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 sm:flex-row sm:items-center"
    >
      <AlertIcon className="h-6 w-6 shrink-0 text-destructive" />
      <p className="flex-1 text-sm text-foreground">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="min-h-[44px] shrink-0 rounded-md bg-destructive px-4 text-sm font-semibold text-destructive-foreground transition hover:brightness-110"
      >
        Tentar de novo
      </button>
    </div>
  );
}

export function IntroState() {
  return (
    <Card>
      <p className="text-[0.95rem] leading-relaxed text-card-foreground">
        A busca lê a transcrição de <strong>todos os sermões</strong> pregados na Igreja
        Presbiteriana Peregrinos. Escreva o que você lembra — um tema, uma passagem, uma frase — e
        nós encontramos o trecho e o link para ouvir.
      </p>
    </Card>
  );
}
