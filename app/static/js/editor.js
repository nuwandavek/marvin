var quillEditor = new Quill('#editor', {
    modules: {
        toolbar: [
            [{ header: [1, 2, false] }],
            ['bold', 'italic', 'underline', 'code'],
            [{ 'color': [] }, { 'background': [] }, { 'font': [] }],
        ]
    },
    placeholder: 'Write away ...',
    theme: 'snow'
});

var quillPreview = new Quill('#preview', {
    modules: {
        toolbar: false
    },
    readOnly: true,
    placeholder: 'No text to preview yet ...',
    theme: 'snow'
});

function RGBAToHexA(r, g, b, a) {
    r = r.toString(16);
    g = g.toString(16);
    b = b.toString(16);
    a = Math.round(a * 255).toString(16);

    if (r.length == 1)
        r = "0" + r;
    if (g.length == 1)
        g = "0" + g;
    if (b.length == 1)
        b = "0" + b;
    if (a.length == 1)
        a = "0" + a;

    return "#" + r + g + b + a;
}

quillEditor.on('text-change', function (delta, oldDelta, source) {
    if (source == 'user') {
        quillPreview.setContents(quillEditor.getContents());
    }
});

function replacePreview(data) {
    let newDelta = {
        ops: []
    }

    var prev_end = 0
    for (var i = 0; i < data.tokens.length; i++) {
        let tempToken = data.tokens[i];
        if (tempToken.start = prev_end + 2) {
            newDelta.ops.push({ insert: ' ' });
        }
        newDelta.ops.push({ insert: tempToken.text, attributes: { background: RGBAToHexA(255, 0, 0, tempToken.attention / 20) } })
    }
    quillPreview.setContents(newDelta);

}

$('#analyze').click(() => {
    let txt = quillPreview.getContents().ops[0].insert;
    $.ajax({
        url: 'http://0.0.0.0:5000/stats',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt },
        success: (d) => {
            console.log(d);
            replacePreview(d);
        }
    });
})