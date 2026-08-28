# Platform adapter contract

Use the independent `jiguang` MCP. Its Windows bridge fixes the origin to `https://jiguang.ascend.huawei.com`, loads the bearer token from Windows Credential Manager, accepts only `/api/` paths selected by typed client methods, and recursively redacts credential-like response fields.

The supported current-account surface is:

- profile;
- local-managed resource pool list/detail/create/update/delete;
- physical-machine and existing-container list/detail/register/delete;
- add a newly registered device to an owned pool;
- deployment list/detail and plan/apply for an existing container;
- evaluation catalog, plan, submit, list, detail, interrupt, and logs/artifacts.

Use `.jiguang/host/set_jiguang_credential.ps1 -Target <name>` for short tokens or passwords. For an existing Windows-side private-key file, use `-SecretFile <path>`; Credential Manager stores only a file reference and the host bridge reads the key without returning it through MCP. Keep that file outside the repository with user-only Windows permissions.

Use `scripts/jiguang_device_key.py --machine <alias> --credential-target <target>` to bind a device to one explicit local SSH identity. It verifies that the selected private key authenticates to the container, reports the matching public-key fingerprint, and stores only a Windows-side private-key file reference. Register the device with `auth_type=SSH_KEY` and that credential target. Container creation receives the derived public-key file explicitly and never copies private-key bytes into the container.

Every mutation has a separate plan or prior read and requires the boolean value `confirm=true`. Resource update/delete calls read the target first and require explicit current-account ownership evidence. Administrator, permission assignment, manager delegation, terminal, plaintext credential, arbitrary HTTP, and arbitrary script operations are absent from the tool catalog.

The host transport connects directly to the internal Jiguang origin and removes inherited HTTP proxy variables for that child process. This avoids routing the internal endpoint through an external enterprise proxy and keeps proxy credentials out of platform error text.

The observed deployment endpoint requires `device_ids`, scenario/software/runtime fields, repository metadata, container mode, and a repository update script. Accept no caller-provided script. Generate the fixed exact-commit checkout script in the policy layer and verify every `device_id` is an existing account-owned container record before POST.

Treat evaluation submit payload compatibility as app-versioned. Always call `evaluation_catalog_list` and `evaluation_plan` first. If the live app rejects the normalized task payload, stop without retrying alternate write endpoints, retain the plan, and update the typed adapter only after the app-specific contract is observed and covered by a fixture test.
