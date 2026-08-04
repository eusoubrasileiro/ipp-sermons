import { type FormEvent, useState } from "react";
import { sendSuggestion } from "../api.ts";

/** Carried over from the old server: a one-line channel for "faltou tal sermão". */
export function SuggestionBox() {
  const [text, setText] = useState("");
  const [sent, setSent] = useState(false);
  const [failed, setFailed] = useState(false);

  async function submit(e: FormEvent): Promise<void> {
    e.preventDefault();
    if (text.trim().length < 3) return;
    try {
      await sendSuggestion(text.trim());
      setSent(true);
      setFailed(false);
      setText("");
    } catch {
      setFailed(true);
    }
  }

  return (
    <section className="mt-10 border-t border-border pt-6">
      <h2 className="text-sm font-semibold">Faltou algum sermão?</h2>
      {sent ? (
        <p className="mt-2 text-sm text-primary">Obrigado! Sua sugestão foi registrada.</p>
      ) : (
        <form onSubmit={submit} className="mt-2 flex flex-col gap-2 sm:flex-row">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Conte para nós qual sermão procurava."
            aria-label="Sugestão"
            className="h-11 w-full rounded-lg border border-input bg-card px-3 text-sm text-card-foreground placeholder:text-muted-foreground focus:border-ring"
          />
          <button
            type="submit"
            className="h-11 shrink-0 rounded-lg border border-border bg-card px-4 text-sm font-medium transition hover:bg-accent hover:text-accent-foreground"
          >
            Enviar
          </button>
        </form>
      )}
      {failed && (
        <p role="alert" className="mt-2 text-sm text-destructive">
          Não foi possível enviar agora. Tente novamente mais tarde.
        </p>
      )}
    </section>
  );
}
