# MCP Client Setup

Use this guide when you want to connect the MCP Slicer Bridge to Claude Code or
Cursor.

## Before You Edit Any Client Config

1. Install [3D Slicer](https://www.slicer.org) and the WebServer extension.
1. Start the Slicer WebServer from `Modules` -> `Developer Tools` -> `WebServer`.
1. Clone this repository and run `uv sync`.
1. Note the absolute path to your local checkout. The MCP config must use that
   absolute path.

Example path:

```text
/Users/you/Documents/3dslicer-claude-bridge
```

## Shared Server Definition

Both clients can use the same stdio server definition:

```json
{
  "mcpServers": {
    "slicer-bridge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/3dslicer-claude-bridge",
        "run",
        "slicer-mcp"
      ],
      "env": {
        "SLICER_URL": "http://localhost:2016"
      }
    }
  }
}
```

If your client config already contains other servers, merge the
`slicer-bridge` entry into the existing `mcpServers` object instead of replacing
the whole file.

## Claude Code

User-scoped Claude Code MCP servers live in `~/.claude.json`.

Add the shared server definition under `mcpServers`, save the file, then
restart Claude Code.

If you prefer project-scoped MCP config, Claude Code also supports a repository
local `.mcp.json`. Use that only if you want the server definition to travel
with the repo.

## Cursor

User-scoped Cursor MCP servers live in `~/.cursor/mcp.json`.

Add the shared server definition, save the file, then restart Cursor. If Cursor
was already open, a full restart is the safest way to force it to reload the MCP
config.

### Discovering Available Capabilities

The server exposes four MCP resources that Cursor can read to discover what is
available without calling any tools:

| Resource | Description |
|----------|-------------|
| `slicer://scene` | Current MRML scene structure and loaded nodes |
| `slicer://volumes` | All loaded imaging volumes with dimensions, spacing, and metadata |
| `slicer://status` | Connection health, Slicer version, and WebServer response time |
| `slicer://workflows` | Catalog of high-level workflow tools with required modalities and clinical indications |

To read a resource in Cursor, ask the agent to fetch it:

```text
Read the slicer://workflows resource and tell me which workflows are available.
```

The `slicer://workflows` resource is especially useful for finding multi-step
clinical pipelines (such as Modic evaluation or oncologic spine assessment) and
understanding which imaging modalities each workflow requires.

## Verification Checklist

After saving the client config:

1. Restart the MCP client.
1. Confirm 3D Slicer is still running with WebServer enabled.
1. Ask the client a simple question such as:

```text
What volumes are loaded in Slicer?
```

1. Try a screenshot request such as:

```text
Show me an axial view of the current volume.
```

If both requests work, the bridge is configured correctly.

## Troubleshooting

### The server does not appear in the client

- Check that the JSON is valid.
- Make sure the server entry is nested under `mcpServers`.
- Restart the client after saving the config.

### The client cannot connect to Slicer

- Confirm Slicer is running.
- Confirm the WebServer extension is installed and started.
- Confirm `SLICER_URL` matches the running WebServer address.

### The MCP server starts, then fails immediately

- Verify the repository path in `--directory` is absolute and points to this
  checkout.
- Run `uv sync` in the repository.
- Reopen the client after fixing the config.

### Requests time out

- Slicer may be busy or frozen. Restart Slicer and try again.
- Confirm the WebServer endpoint is still reachable.

### You changed the config, but the client still uses old settings

- Fully restart Claude Code or Cursor.
- Double-check that you edited the correct user config file.

## Related Docs

- Return to the main onboarding guide in [`README.md`](../README.md).
- Contributor setup and test commands are documented in
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).
