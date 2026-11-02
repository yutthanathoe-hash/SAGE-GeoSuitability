# Publish v1.0.0 on GitHub

1. Review and approve `LICENSE` (proposed MIT), `CITATION.cff`, and the example data for public sharing.
2. Create a new repository named `SAGE-GeoSuitability` in your GitHub account. Do not initialize with a README if pushing this folder.
3. From the unzipped repository root run:

```bash
git init
git add .
git commit -m "Release SAGE-GeoSuitability v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/SAGE-GeoSuitability.git
git push -u origin main
git tag -a v1.0.0 -m "SAGE-GeoSuitability v1.0.0"
git push origin v1.0.0
```

4. GitHub → Releases → Draft a new release → select tag `v1.0.0`; title `SAGE-GeoSuitability v1.0.0`; paste `RELEASE_NOTES_v1.0.0.md`. Attach the release ZIP if desired.
5. Publish the release only after reviewing license and scientific caveats. Optionally connect Zenodo after publication.
