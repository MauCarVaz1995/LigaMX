"""
20_twitter_bot.py — Bot de publicación Twitter para @Miau_Stats_MX
Usa Twitter API v2 con OAuth 1.0a (necesario para subir media).

Credenciales requeridas (variables de entorno):
    TWITTER_API_KEY          ← Consumer Key
    TWITTER_API_SECRET       ← Consumer Secret
    TWITTER_ACCESS_TOKEN     ← Access Token
    TWITTER_ACCESS_SECRET    ← Access Token Secret

Uso:
    from scripts.20_twitter_bot import publicar_prediccion, publicar_hilo

    publicar_prediccion("output/charts/pred_CZA_PAC.png", "Cruz Azul vs Pachuca...")
    publicar_hilo([
        {"imagen": "pred_CZA_PAC.png", "texto": "J14 — Cruz Azul vs Pachuca 🔵"},
        {"imagen": "pred_MTY_SLP.png", "texto": "J14 — Monterrey vs San Luis ⚽"},
    ])
"""

import os
import sys
import tweepy


# ── Credenciales ──────────────────────────────────────────────────────────────

def _get_client():
    """Construye cliente Tweepy v2 con OAuth 1.0a. Lanza ValueError si faltan credenciales."""
    required = [
        "TWITTER_API_KEY",
        "TWITTER_API_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_SECRET",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise ValueError(
            f"Faltan variables de entorno: {', '.join(missing)}\n"
            "Exporta las 4 credenciales antes de usar el bot:\n"
            "  export TWITTER_API_KEY=...\n"
            "  export TWITTER_API_SECRET=...\n"
            "  export TWITTER_ACCESS_TOKEN=...\n"
            "  export TWITTER_ACCESS_SECRET=..."
        )

    client = tweepy.Client(
        consumer_key=os.environ["TWITTER_API_KEY"],
        consumer_secret=os.environ["TWITTER_API_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_SECRET"],
    )
    return client


def _get_api_v1():
    """Construye API v1.1 (necesaria para subir media — upload_media no está en v2)."""
    auth = tweepy.OAuth1UserHandler(
        consumer_key=os.environ["TWITTER_API_KEY"],
        consumer_secret=os.environ["TWITTER_API_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_SECRET"],
    )
    return tweepy.API(auth)


# ── Validación de credenciales ────────────────────────────────────────────────

def validar_credenciales(verbose=True):
    """
    Verifica que las credenciales conectan correctamente con la API.
    NO publica nada. Devuelve True si OK, False si falla.
    """
    try:
        client = _get_client()
        me = client.get_me()
        if me.data:
            if verbose:
                print(f"✅ Conexión exitosa")
                print(f"   Usuario : @{me.data.username}")
                print(f"   Nombre  : {me.data.name}")
                print(f"   ID      : {me.data.id}")
            return True
        else:
            if verbose:
                print("❌ La API respondió pero no devolvió datos de usuario.")
            return False
    except tweepy.errors.Unauthorized as e:
        if verbose:
            print(f"❌ Credenciales inválidas (401 Unauthorized): {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"❌ {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Error inesperado: {e}")
        return False


# ── Subida de media ───────────────────────────────────────────────────────────

def _subir_imagen(api_v1, imagen_path):
    """Sube una imagen PNG a Twitter y devuelve el media_id."""
    if not os.path.isfile(imagen_path):
        raise FileNotFoundError(f"No existe el archivo: {imagen_path}")
    media = api_v1.media_upload(filename=imagen_path)
    return media.media_id


# ── Funciones principales ─────────────────────────────────────────────────────

def publicar_prediccion(imagen_path, texto, dry_run=False):
    """
    Publica un tweet con una imagen PNG y texto.

    Args:
        imagen_path : ruta al PNG (absoluta o relativa al directorio de trabajo)
        texto       : texto del tweet (máx 280 caracteres)
        dry_run     : si True, NO publica — solo valida y muestra lo que haría

    Returns:
        tweet_id (str) si se publicó, None si dry_run o error
    """
    if len(texto) > 280:
        print(f"⚠️  El texto tiene {len(texto)} caracteres (máx 280). Truncando.")
        texto = texto[:277] + "..."

    if dry_run:
        print("── DRY RUN ──────────────────────────────────")
        print(f"  Imagen : {imagen_path}")
        print(f"  Texto  : {texto}")
        print(f"  Chars  : {len(texto)}/280")
        print("  Acción : NO publicado (dry_run=True)")
        print("─────────────────────────────────────────────")
        return None

    client = _get_client()
    api_v1 = _get_api_v1()

    media_id = _subir_imagen(api_v1, imagen_path)
    response = client.create_tweet(text=texto, media_ids=[media_id])
    tweet_id = response.data["id"]
    print(f"✅ Tweet publicado: https://twitter.com/i/web/status/{tweet_id}")
    return tweet_id


def publicar_hilo(partidos, dry_run=False):
    """
    Publica una lista de tweets como hilo — cada uno en respuesta al anterior.

    Args:
        partidos : lista de dicts con claves:
                     "imagen" → ruta al PNG
                     "texto"  → texto del tweet
        dry_run  : si True, NO publica — solo muestra lo que haría

    Returns:
        lista de tweet_ids publicados (vacía si dry_run)
    """
    if not partidos:
        print("⚠️  Lista de partidos vacía.")
        return []

    if dry_run:
        print(f"── DRY RUN HILO ({len(partidos)} tweets) ──────────────────")
        for i, p in enumerate(partidos, 1):
            chars = len(p["texto"])
            estado = "↩ respuesta al anterior" if i > 1 else "✦ tweet inicial"
            print(f"  [{i}] {estado}")
            print(f"       Imagen : {p['imagen']}")
            print(f"       Texto  : {p['texto'][:80]}{'...' if chars > 80 else ''}")
            print(f"       Chars  : {chars}/280")
        print("  Acción : NO publicado (dry_run=True)")
        print("──────────────────────────────────────────────────")
        return []

    client = _get_client()
    api_v1 = _get_api_v1()

    tweet_ids = []
    reply_to = None

    for i, partido in enumerate(partidos, 1):
        texto = partido["texto"]
        if len(texto) > 280:
            texto = texto[:277] + "..."

        media_id = _subir_imagen(api_v1, partido["imagen"])

        kwargs = {"text": texto, "media_ids": [media_id]}
        if reply_to:
            kwargs["in_reply_to_tweet_id"] = reply_to

        response = client.create_tweet(**kwargs)
        tweet_id = response.data["id"]
        tweet_ids.append(tweet_id)
        reply_to = tweet_id
        print(f"  [{i}/{len(partidos)}] ✅ https://twitter.com/i/web/status/{tweet_id}")

    print(f"\n✅ Hilo publicado ({len(tweet_ids)} tweets)")
    return tweet_ids


# ── CLI / test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Twitter bot — MAU-STATISTICS")
    subparsers = parser.add_subparsers(dest="cmd")

    # Validar credenciales
    subparsers.add_parser("validar", help="Verifica que las credenciales conectan")

    # Test dry-run
    p_test = subparsers.add_parser("test", help="Test dry-run sin publicar")
    p_test.add_argument("--imagen", default=None, help="Ruta al PNG (opcional)")

    # Publicar predicción individual
    p_pub = subparsers.add_parser("publicar", help="Publica un tweet con imagen")
    p_pub.add_argument("imagen", help="Ruta al PNG")
    p_pub.add_argument("texto", help="Texto del tweet")
    p_pub.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.cmd == "validar":
        validar_credenciales()

    elif args.cmd == "test":
        # Usa imagen dummy si no se especifica una real
        imagen = args.imagen
        if not imagen:
            # Busca cualquier PNG en output/charts/predicciones/
            import glob as _glob
            pngs = _glob.glob("output/charts/predicciones/**/*.png", recursive=True)
            if pngs:
                imagen = pngs[0]
            else:
                pngs = _glob.glob("output/charts/**/*.png", recursive=True)
                imagen = pngs[0] if pngs else "dummy.png"

        texto_test = "Test @Miau_Stats_MX — bot funcionando ⚽📊 #LigaMX #MauStats"
        print(f"\n📋 PASO 1 — Validando credenciales...")
        ok = validar_credenciales(verbose=True)

        print(f"\n📋 PASO 2 — Dry-run de publicar_prediccion()...")
        publicar_prediccion(imagen, texto_test, dry_run=True)

        print(f"\n📋 PASO 3 — Dry-run de publicar_hilo() (2 tweets)...")
        publicar_hilo([
            {"imagen": imagen, "texto": texto_test},
            {"imagen": imagen, "texto": "Continuación del hilo 👆 | Más predicciones disponibles en @Miau_Stats_MX"},
        ], dry_run=True)

    elif args.cmd == "publicar":
        publicar_prediccion(args.imagen, args.texto, dry_run=args.dry_run)

    else:
        parser.print_help()
