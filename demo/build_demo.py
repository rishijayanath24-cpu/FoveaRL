"""Bundle template + css + three.js + data + app into one self-contained HTML file."""
import os
H = os.path.dirname(os.path.abspath(__file__))
r = lambda p: open(os.path.join(H, p), encoding="utf-8").read()
three = r("vendor_dl/three.min.js")
three = three.replace("</script", "<\\/script")      # safety for inlining
html = (r("src/template.html")
        .replace("/*__CSS__*/", r("src/style.css"))
        .replace("/*__THREE__*/", three)
        .replace("/*__DATA__*/", r("data.js"))
        .replace("/*__APP__*/", r("src/app.js")))
out = os.path.join(H, "FoveaRL_demo.html")
open(out, "w", encoding="utf-8").write(html)
print(out, round(len(html) / 1e6, 2), "MB")
