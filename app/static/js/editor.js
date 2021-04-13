var result = null;

var dropdownSelection = null;

function storeResult(data) {
    result = data;
}

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

function setSliders(data) {
    $('#formality-slider').slider('set value', data.formality.prob * 100);
    $('#humor-slider').slider('set value', data.jokes.prob * 100);

    $('#formality-value').html((data.formality.prob * 100).toFixed(2));
    $('#humor-value').html((data.jokes.prob * 100).toFixed(2));
}

function displayHeatmap(data) {
    output_txt = ''
    recent_end = 0
    for (let i = 0; i < data.tokens.length; i++) {
        let currToken = data.tokens[i]
        if (recent_end != currToken.start) {
            output_txt += " ";
        }
        output_txt += "<span style='background : " + RGBAToHexA(255, 0, 0, currToken.attention / 20) + "'>" + currToken.text + "</span>"
    }
    $('#preview').html(output_txt);
}

function displayJointHeatmap(data) {
    output_txt = ''
    recent_end = 0
    let query = null;
    if (dropdownSelection === 'formality') {
        query = 'formality'
    }
    else if (dropdownSelection === 'humor') {
        query = 'jokes'
    }

    if (query != null) {
        for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
                output_txt += " ";
                recent_end += 1;
            }
            output_txt += "<span style='background : " + RGBAToHexA(255, 0, 0, data[query]['salience'][i] / 2) + "'>" + currToken.text + "</span>"
            recent_end += currToken.text.length
        }
        $('#preview').html(output_txt);
    }
    else {
        $('#preview').html(data.input);
    }

}


$('#analyze').click(() => {
    let txt = quillEditor.getContents().ops[0].insert;
    $.ajax({
        url: 'http://0.0.0.0:5000/stats',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt },
        success: (d) => {
            storeResult(d);
            // displayHeatmap(d.attn);
            setSliders(d.joint);
            displayJointHeatmap(d.joint);

        }
    });
})



$('#formality-slider').slider({
    min: 0,
    max: 100,
    start: 0,
    step: 1
});

$('#humor-slider').slider({
    min: 0,
    max: 100,
    start: 0,
    step: 1
});

$('.dropdown')
    .dropdown({
        onChange: function (value, text, $selectedItem) {
            console.log(value);
            dropdownSelection = value;
        }
    })
    ;