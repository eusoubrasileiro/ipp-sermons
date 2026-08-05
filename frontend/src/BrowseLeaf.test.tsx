import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { routeTo } from "./browse-fixtures.tsx";
import { FACETS } from "./facet-fixtures.ts";
import { okResponse } from "./test-fixtures.tsx";

/**
 * The leaf pages an index links to, and what they ask the server for.
 */
describe("folhas restantes", () => {
  it("mostra a série-mãe como segunda linha no índice", async () => {
    routeTo("/series");
    // It is both a series of its own and the parent shown under CFW 3.
    const mentions = await screen.findAllByText("Confissão de Fé de Westminster");
    expect(mentions.length).toBeGreaterThan(1);
  });

  it("abre um curso em ordem de aula", async () => {
    routeTo("/series/cfw-3");
    expect(await screen.findByRole("heading", { name: /CFW 3/ })).toBeInTheDocument();
    const calls = (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls;
    const listing = calls.map((c) => c[0]).find((u) => u?.startsWith("/api/sermons"));
    expect(listing).toContain("ordenar=serie");
  });

  it("abre a página de um pregador com o cargo no título", async () => {
    routeTo("/pregadores/reverendo-bruno-melo");
    expect(
      await screen.findByRole("heading", { name: "Reverendo Bruno Melo" }),
    ).toBeInTheDocument();
  });

  it("abre um ano inteiro quando o mês não é dito", async () => {
    routeTo("/datas/2024");
    await screen.findByRole("heading", { name: "2024" });
    const calls = (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls;
    const listing = calls.map((c) => c[0]).find((u) => u?.startsWith("/api/sermons"));
    expect(listing).toContain("de=2024-01-01");
    expect(listing).toContain("ate=2024-12-31");
  });

  it("mostra a descrição quando a série não tem mãe", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.startsWith("/api/facets")
          ? okResponse({
              ...FACETS,
              series: [
                {
                  slug: "diaconia",
                  nome: "Diaconia",
                  kind: "diaconia",
                  paiSlug: null,
                  descricao: "Curso sobre o ministério dos diáconos.",
                  total: 7,
                },
              ],
            })
          : okResponse({ total: 0, sermons: [], pagina: 1 }),
      ),
    );
    routeTo("/series");
    expect(await screen.findByText("Curso sobre o ministério dos diáconos.")).toBeInTheDocument();
  });

  it("diz que o índice está vazio em vez de mostrar nada", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => okResponse({ ...FACETS, pregadores: [] })),
    );
    routeTo("/pregadores");
    expect(await screen.findByText(/Nada por aqui ainda/)).toBeInTheDocument();
  });

  it("cai numa folha desconhecida sem quebrar", async () => {
    routeTo("/biblia/livro-que-nao-existe");
    expect(
      await screen.findByRole("heading", { name: "livro-que-nao-existe" }),
    ).toBeInTheDocument();
  });
});

describe("estados", () => {
  it("avisa quando os índices não carregam", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, json: async () => ({}) }) as Response),
    );
    routeTo("/biblia");
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("volta para o índice a partir de uma folha", async () => {
    routeTo("/biblia/efesios/5");
    await userEvent.click(await screen.findByRole("link", { name: /← Bíblia/ }));
    expect(await screen.findByText("Antigo Testamento")).toBeInTheDocument();
  });

  it("diz quando a faceta está vazia", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.startsWith("/api/facets")
          ? okResponse(FACETS)
          : okResponse({ total: 0, sermons: [], pagina: 1 }),
      ),
    );
    routeTo("/biblia/efesios/5");
    await waitFor(() => expect(screen.getByText(/Nenhum sermão aqui ainda/)).toBeInTheDocument());
  });
});
