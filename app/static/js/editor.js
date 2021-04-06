var result = null;

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
    let formalProb = 0;
    let humorProb = 0;
    if (data.formality.class === 'formal') {
        formalProb = data.formality.prob;
    }
    else if (data.formality.class === 'informal') {
        formalProb = 1 - data.formality.prob
    }
    if (data.jokes.class === 'joke') {
        humorProb = data.jokes.prob;
    }
    else if (data.jokes.class === 'nojoke') {
        humorProb = 1 - data.jokes.prob
    }
    $('#formality-slider').slider('set value', formalProb * 100);
    $('#humor-slider').slider('set value', humorProb * 100);

    $('#formality-value').html((formalProb * 100).toFixed(2));
    $('#humor-value').html((humorProb * 100).toFixed(2));
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


function displayFormalSalienceHeatmap(data) {
    if (data != null) {
	output_txt = ''
	recent_end = 0
	// class_label = data.formality.class
	class_label = 'formal';
	for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
		output_txt += " ";
            }
            output_txt += "<span style='background : " + RGBAToHexA(0, 255, 0, data.formality.salience[class_label][i+1]*10) + "'>" + currToken.text + "</span>"
	}
	$('#preview').html(output_txt);
    }
}


function displayHumorSalienceHeatmap(data) {
    if (data != null) {
	output_txt = ''
	recent_end = 0
	// class_label = data.jokes.class
	class_label = 'joke';
	for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
		output_txt += " ";
            }
            output_txt += "<span style='background : " + RGBAToHexA(0, 0, 255, data.jokes.salience[class_label][i+1]) + "'>" + currToken.text + "</span>"
	}
	$('#preview').html(output_txt);
    }
}


function displayInformalSalienceHeatmap(data) {
    if (data != null) {
	output_txt = ''
	recent_end = 0
	// class_label = data.formality.class
	class_label = 'informal';
	for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
		output_txt += " ";
            }
            output_txt += "<span style='background : " + RGBAToHexA(0, 255, 0, data.formality.salience[class_label][i+1]*10) + "'>" + currToken.text + "</span>"
	}
	$('#preview').html(output_txt);
    }
}


function displayNonHumorSalienceHeatmap(data) {
    if (data != null) {
	output_txt = ''
	recent_end = 0
	// class_label = data.jokes.class
	class_label = 'nojoke';
	for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
		output_txt += " ";
            }
            output_txt += "<span style='background : " + RGBAToHexA(0, 0, 255, data.jokes.salience[class_label][i+1]) + "'>" + currToken.text + "</span>"
	}
	$('#preview').html(output_txt);
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
            displayHeatmap(d.attn);
            setSliders(d.joint);
        }
    });
})


$('#formal-salience').click(() => {
    displayFormalSalienceHeatmap(result.joint);
})

$('#humor-salience').click(() => {
    displayHumorSalienceHeatmap(result.joint);
})

$('#informal-salience').click(() => {
    displayInformalSalienceHeatmap(result.joint);
})

$('#non-humor-salience').click(() => {
    displayNonHumorSalienceHeatmap(result.joint);
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
