# Public hosting plan for RetinaCare

## Netlify static front-end

This project is not a pure static app. The browser UI in `web_ui/index.html` depends on the Flask API in `web_ui/server.py` for login, history, personal guide, and image screening. Netlify can host the front end, but it cannot run this Python server directly.

## Recommended deployment pattern

1. Deploy the static UI to Netlify.
2. Deploy the Python API to a service that supports Flask, such as Render, Railway, Fly.io, or a VPS.
3. Set the public API URL in the Netlify app as a build environment variable or by editing the page script before deploy.

## Netlify env variable to set

Set this in the Netlify UI:

`RETINACARE_API_URL=https://your-python-api-domain.example`

The front end will use it automatically via the `window.RETINACARE_API_URL` variable.

## Example Flask API deployment

The app already includes a `render.yaml` file for Render, and the Python app can be started with:

```bash
gunicorn --chdir web_ui --workers 2 --threads 4 --timeout 120 server:app
```

## Important note

This project includes MATLAB/AI pipeline code and a development medical baseline. The browser flow is a demo interface, not a real clinical system. Use a secure backend and proper authentication before public deployment.
