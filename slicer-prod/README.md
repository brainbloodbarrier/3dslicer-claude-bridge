# slicer-prod

Superfície enxuta de produção para usar o `slicer-mcp` com **Claude Code** e
**OpenCode** sem carregar a árvore completa de desenvolvimento como contexto de
trabalho principal.

## O que este diretório faz

- expõe o servidor MCP do projeto via configuração local de cliente;
- mantém os **comandos do Claude** sempre apontando para a origem canônica;
- gera os **comandos equivalentes do OpenCode** a partir da mesma fonte;
- evita duplicar `src/`, `tests/`, `docs/` e outros recursos de desenvolvimento.

## Fonte da verdade

- Código MCP: `../src/slicer_mcp`
- Entry point: `../pyproject.toml` (`uv run slicer-mcp`)
- Claude commands: `../.claude/commands/*.md`

Se você adicionar novas tools, workflows ou resources no código principal, o
servidor daqui já enxerga isso automaticamente porque o comando MCP aponta para
o repositório pai.

Se você editar ou criar comandos em `../.claude/commands`, rode:

```bash
python3 slicer-prod/scripts/sync_surface.py
```

Isso atualiza `slicer-prod/.opencode/commands` e valida que a superfície de
produção continua sincronizada com a quantidade e o conteúdo atuais dos comandos
canônicos.

## Estrutura

```text
slicer-prod/
├── .claude/
│   ├── commands -> ../../.claude/commands
│   └── settings.local.json
├── .mcp.json
├── .opencode/
│   └── commands/                # gerado a partir dos Claude commands
├── CLAUDE.md
├── opencode.json
└── scripts/
    └── sync_surface.py
```

## Claude Code

- abra a sessão a partir de `slicer-prod/`;
- o servidor MCP do projeto é carregado por `./.mcp.json`;
- os comandos disponíveis vêm do symlink `./.claude/commands`.

## OpenCode

- abra a sessão a partir de `slicer-prod/`;
- a config do projeto está em `./opencode.json`;
- os comandos customizados vivem em `./.opencode/commands` e são gerados a
  partir dos comandos do Claude.

## Resources expostos pelo servidor

- `slicer://scene`
- `slicer://volumes`
- `slicer://status`
- `slicer://workflows`

## Regra de manutenção

Não edite `slicer-prod/.opencode/commands/*.md` manualmente. Eles são gerados.
Edite apenas `../.claude/commands/*.md` e rode o sincronizador.
