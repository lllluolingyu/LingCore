# Lark / Feishu channel

LingCore supports conversational bots in Feishu (China) and Lark
(international), using the official `lark-channel-sdk==1.4.0`. The channel
opens an outbound WebSocket connection, so no public HTTP listener, webhook
verification token, or reverse proxy is needed. See the official
[SDK quickstart](https://github.com/larksuite/channel-sdk-python/blob/main/docs/quickstart.md)
for the connection setup and domain selection.

## Configure the application

1. Create a custom app in the [Feishu developer console](https://open.feishu.cn/app)
   or [Lark developer console](https://open.larksuite.com/app), and enable its
   bot capability.
2. Grant `im:message:send_as_bot` to send replies, and
   `im:message.p2p_msg:readonly` to receive direct messages. For groups, also
   grant `im:message.group_at_msg:readonly`; for images and files, grant
   `im:resource`. The platform documents these in its
   [messaging APIs](https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1)
   and [bot permissions](https://open.feishu.cn/document/develop-robots/add-bot-to-external-group?lang=zh-CN).
3. Under Events & Callbacks, select **Use long connection to receive events**
   and subscribe to **Receive message v2.0** (`im.message.receive_v1`).
   Start LingCore with the credentials below if the console requires an active
   connection before saving this setting.
4. Publish the app version and make it available to the intended users.
   Apply/publish permission changes before testing, and add the bot to any
   groups you intend to enable.

## Install and run

```bash
pip install "lingcore[lark]"
cp lark.yaml.example profiles/coding/lark.yaml
```

For a source checkout, `uv sync --extra lark` installs the integration.
`lingcore[feishu]` is an alias for the same optional dependency. The base
package does not import or require the SDK.

Put credentials in the selected profile's `.env`, alongside its existing
model settings, or supply them through the process environment:

```dotenv
LARK_APP_ID=cli_your_app_id
LARK_APP_SECRET=your_app_secret
```

`app_id_env` and `app_secret_env` name variables; they do not hold credentials.
Profile `.env` entries take precedence over process variables, including an
explicitly empty entry. Diagnostics do not print secret values.

In `lark.yaml`, set `domain: feishu` for Feishu or `domain: lark` for Lark,
and replace the placeholder in `allowed_user_ids` with your app-scoped
`open_id` (`ou_…`). An unlisted user can send a direct message to obtain this
ID from the bot without creating an Agent; add it to the file and restart.
Keep at least one entry in the allowlist, including during setup.

```bash
lingcore doctor --profile profiles/coding \
  --lark-config profiles/coding/lark.yaml
lingcore lark --profile profiles/coding
# Equivalent command:
lingcore feishu --profile profiles/coding
```

Both commands read `<profile>/lark.yaml`. `--lark-config <path>` and
`--feishu-config <path>` select another file. The configured `domain`, not the
command alias, selects which service is used. `doctor` validates the channel
configuration and required environment variables offline, without the SDK.

Groups are disabled unless their `chat_id` (`oc_…`) is in `allowed_chat_ids`.
In an enabled group, the sender must also be in `allowed_user_ids` and must
mention the bot. The same rule applies to commands and approvals. Bot/app
senders are ignored. Group replies and tool approval requests are visible
to the group's members.

## Chat behavior

| Input | Behavior |
| --- | --- |
| Text or rich-text post | Start an agent turn; the bot's own mention is removed |
| One image or file | Download and ingest as an attachment; images up to 5 MiB, files up to 10 MiB |
| `/help` | Show commands, including profile/plugin commands |
| `/new` | Start a new session in this conversation |
| `/sessions` | List the ten most recent sessions in this conversation |
| `/resume <id-prefix>` | Switch to a stored session |
| `/stop` | Cancel the active turn and deny pending tool approvals |
| `/approve <token>` / `/deny <token>` | Answer a pending tool approval |

Only one turn runs per conversation/sender pair. A new prompt during a turn
gets a busy response; use `/stop` first. Session changes are refused while a
turn is active. CLI session/workspace flags are not accepted by the channel.

Responses are sent as plain text once the final output guardrail has run.
Incomplete responses use the guarded partial output. Long replies are split
into messages; generated image/file attachments are also delivered. This
version does not stream tokens, handle card callbacks, accept voice/video,
expand forwarded messages, or ingest multiple attachments in one message.
The SDK downloads incoming media into memory before LingCore applies its
attachment size and content checks.

Approvals use random, one-use tokens and are bound to the requesting user
and chat. They expire after `confirmation_timeout` seconds and are denied on
Stop or shutdown. Shell tools require `require_confirmation: true` and an
empty `allow_patterns` list. Background process tools require an attended
frontend and are refused here.

## State and operation

Each chat/sender pair has its own workspace, sessions, memory, calendar,
traces, and mutable tool options. A direct conversation and a group conversation
with the same user do not share agent history. State is stored under:

```text
<profile>/.lingcore/lark/<domain>/
  bridge.sqlite3
  conversations/<hash-of-chat-and-user>/
    sessions.db
    workspace/
    memory.md
```

The active session selection survives restarts. Invalid stored selections
are replaced after a new runtime builds successfully. `state_dir` is confined
to the profile unless an absolute path is explicitly enabled with
`allow_absolute_state_dir: true`; package directories are always refused.
Use a separate state directory for each bot application.

Run one channel process per app/state directory. Message deduplication is
bounded and in memory; it does not provide exactly-once processing across
restarts. The SDK owns reconnects and outbound retries. A delivery failure is
logged without undoing a completed agent turn; the transcript remains stored.

If messages do not arrive, check app publication and availability, event
subscriptions, permissions, domain selection, sender/group allowlists, and
the bot mention. The host needs HTTPS access to the selected API domain and
WSS access to the WebSocket endpoint returned by that service. The
[SDK reference](https://github.com/larksuite/channel-sdk-python/blob/main/docs/reference.md)
describes its transport and send-result contracts.
