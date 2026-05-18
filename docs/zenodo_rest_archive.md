# Zenodo REST API Archive Workflow

This workflow creates a Zenodo software deposition directly through the REST API.
Use the sandbox first; production publication is permanent.

## 1. Create a Zenodo Token

Create a token on the same Zenodo environment you plan to use:

- Sandbox: https://sandbox.zenodo.org/account/settings/applications/tokens/new/
- Production: https://zenodo.org/account/settings/applications/tokens/new/

Select these scopes:

- `deposit:write`
- `deposit:actions`

Export it locally:

```bash
export ZENODO_TOKEN="paste-token-here"
```

## 2. Commit the Archive Metadata

The upload script uses `git archive HEAD`, so only committed files are included in
the ZIP archive. Commit `zenodo_metadata.json`, this guide, and the upload script
before doing the production upload.

## 3. Test on Sandbox

```bash
python scripts/zenodo_deposit.py --dry-run --keep-archive

python scripts/zenodo_deposit.py \
  --base-url https://sandbox.zenodo.org \
  --keep-archive
```

This creates a draft sandbox deposition but does not publish it.

## 4. Production Draft

After reviewing the sandbox result, create the production draft:

```bash
python scripts/zenodo_deposit.py \
  --base-url https://zenodo.org \
  --keep-archive
```

Open the draft URL printed by the script and review the metadata, files, and
reserved DOI.

If draft creation succeeds but file upload fails, retry the upload against the
existing draft:

```bash
python scripts/zenodo_deposit.py \
  --base-url https://zenodo.org \
  --keep-archive \
  --upload-existing DEPOSITION_ID
```

If metadata changes after draft creation, update the existing draft:

```bash
python scripts/zenodo_deposit.py \
  --base-url https://zenodo.org \
  --update-existing DEPOSITION_ID
```

## 5. Publish

When the production draft looks correct, publish that reviewed draft by ID:

```bash
python scripts/zenodo_deposit.py \
  --base-url https://zenodo.org \
  --publish-existing DEPOSITION_ID
```

Publishing creates the public Zenodo record and registers the DOI. Published
depositions cannot be deleted.

If you intentionally want to create, upload, and publish in one command, use:

```bash
python scripts/zenodo_deposit.py \
  --base-url https://zenodo.org \
  --keep-archive \
  --publish
```

## Metadata Sources

The software archive is linked to:

- Article DOI: https://doi.org/10.1038/s41398-026-04091-6
- GitHub repository: https://github.com/ftlabucsd/FED3-data
