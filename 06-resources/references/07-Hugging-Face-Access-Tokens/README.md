# Hugging Face Access Tokens — Secure Authentication Guide

Hugging Face user access tokens authenticate scripts, notebooks, command-line tools, and applications. They are credentials: anyone possessing a valid token may perform the actions allowed by its scope.

## Resource Summary

| Item | Details |
|---|---|
| Management page | <https://huggingface.co/settings/tokens> |
| Official guide | <https://huggingface.co/docs/hub/security-tokens> |
| Course association | Session 14 · `07_a_sam3_huggingface` |
| Recommended course scope | Read or fine-grained access to the required gated model |

## Token Roles

| Role | Capability | Appropriate use |
|---|---|---|
| Read | Download repositories the account may read | Course model download and inference |
| Write | Read plus create or modify repositories | Uploading models or datasets |
| Fine-grained | Explicit permissions for selected resources | Production and least-privilege access |

Do not create a write token when the task only downloads a model.

## Create a Token

1. Sign in at <https://huggingface.co/>.
2. Open <https://huggingface.co/settings/tokens>.
3. Select **New token**.
4. Give it a purpose-specific name such as `sam3-colab-read`.
5. Select read or fine-grained access.
6. Copy the token once and store it securely.

Use a separate token for a local computer, Colab, CI, and production. This allows one credential to be revoked without disrupting every environment.

## Local Authentication

Install the client:

```bash
python -m pip install -U huggingface_hub
```

Authenticate interactively:

```bash
hf auth login
```

Check the authenticated account:

```bash
hf auth whoami
```

Log out:

```bash
hf auth logout
```

The interactive prompt masks the token. Avoid including a raw credential directly in a shell command because it may be stored in command history.

## Google Colab Secret

In Colab, open the Secrets panel, create a secret named `HF_TOKEN`, paste the value there, and enable notebook access.

```python
from google.colab import userdata
from huggingface_hub import login

token = userdata.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is missing from Colab Secrets")

login(token=token)
```

Never display `token` or include it in an exception.

## Environment Variable

Local terminal:

```bash
export HF_TOKEN="your-token-from-a-secure-secret-store"
```

Windows PowerShell for the current session:

```powershell
$env:HF_TOKEN = "your-token-from-a-secure-secret-store"
```

Python libraries can detect `HF_TOKEN`. Do not place the actual value in `.env.example`; only document the variable name.

## Gated Models

A token does not automatically grant access to a gated model.

```text
Hugging Face account
       ↓
Accept model conditions / request access
       ↓
Approval when required
       ↓
Create suitable token
       ↓
Authenticate the environment
       ↓
Download permitted files
```

For `facebook/sam3`, open the model page using the same account represented by the token.

## Safe Verification

```python
from huggingface_hub import whoami

profile = whoami()
print(profile["name"])
```

This verifies the account without printing the token.

## Never Commit These

```text
HF_TOKEN=hf_actual_secret_value
token = "hf_actual_secret_value"
Authorization: Bearer hf_actual_secret_value
```

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
*.token
secrets/
```

An `.env.example` may safely contain:

```text
HF_TOKEN=
```

## If a Token Leaks

1. Revoke or refresh it immediately in token settings.
2. Remove it from notebooks, logs, screenshots, commits, and build artifacts.
3. If committed, treat repository history as exposed even after deleting the visible line.
4. Create a replacement with the smallest necessary permissions.
5. Review account and repository activity.

Do not merely rename or hide the leaked token.

## Error Guide

| Error | Meaning | Action |
|---|---|---|
| `401 Unauthorized` | Missing, invalid, expired, or revoked token | Authenticate again with a valid token |
| `403 Forbidden` | Valid identity lacks permission | Accept model terms, wait for approval, or correct token scope |
| `RepositoryNotFoundError` | Wrong model ID or inaccessible gated/private repository | Verify spelling, account, access, and authentication |
| Colab secret returns `None` | Secret missing or notebook access disabled | Check the exact name and enable access |
| GitHub secret scanning alert | Credential was committed | Revoke immediately and clean the exposed location |

## Professional Security Practices

- Follow least privilege.
- Use one token per environment or purpose.
- Prefer fine-grained tokens in production.
- Rotate credentials periodically and after personnel or access changes.
- Use platform secrets rather than plaintext files.
- Keep tokens out of screenshots and copied error messages.
- Never ask another learner to share a personal token.
- Do not publish model files when redistribution is restricted.

## Official References

- Token management: <https://huggingface.co/settings/tokens>
- User access tokens: <https://huggingface.co/docs/hub/security-tokens>
- Gated models: <https://huggingface.co/docs/hub/models-gated>
- Hub authentication: <https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login>
