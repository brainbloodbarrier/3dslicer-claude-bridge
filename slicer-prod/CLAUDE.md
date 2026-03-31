# CLAUDE.md

Este diretório é uma superfície mínima de uso do `slicer-mcp`.

## Escopo

- Use este diretório quando quiser trabalhar só com o servidor MCP do 3D Slicer.
- A fonte da verdade do código continua no repositório pai, em `../src/slicer_mcp`.
- A fonte da verdade dos comandos continua em `../.claude/commands`.

## Regras locais

- Não adicione skills, plugins ou comandos genéricos aqui.
- Não edite `./.opencode/commands/*.md` manualmente; rode
  `python3 slicer-prod/scripts/sync_surface.py`.
- Se precisar alterar o comportamento do servidor, mude o código em
  `../src/slicer_mcp` e não neste diretório.

## Configuração MCP

- Claude Code usa `./.mcp.json`.
- OpenCode usa `./opencode.json`.
- Ambos executam `uv --directory .. run slicer-mcp`, então qualquer nova tool ou
  resource registrada no projeto principal já aparece aqui.
