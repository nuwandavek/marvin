import { RGBAToHexA } from "./utils.js";

function setProgress(data, mode) {
    console.log(mode, data);
    if (mode === "micro-formality") {
        $('#formality-progress-' + mode)
            .progress({
                duration: 200,
                percent: parseInt(data.formality.prob * 100),
                text: {
                    active: 'Formality : {percent} %'
                }
            });
    }
    else if (mode === "micro-joint") {
        $('#formality-progress-' + mode)
            .progress({
                duration: 200,
                percent: parseInt(data.formality.prob * 100),
                text: {
                    active: 'Formality : {percent} %'
                }
            });
        $('#emo-progress-' + mode)
            .progress({
                duration: 200,
                percent: parseInt(data.emo.prob * 100),
                text: {
                    active: 'Emotion : {percent} %'
                }
            });
    }
    else if (mode === "macro-shakespeare") {
        $('#shakespeare-progress-' + mode)
            .progress({
                duration: 200,
                percent: parseInt(data.shakespeare.prob * 100),
                text: {
                    active: 'Shakespeare : {percent} %'
                }
            });

    }


}

function displayModal(data, mode) {
    let modalHTML = '';

    let originalText = "<div class='row middle-xs'>" +
        "<p class='modal-text'><span class='ui label'>Original Text</span><span class='original'>" + data.input.text + "</span></p>"
    if (mode === "micro-formality") {
        originalText += "<div class='ui teal image label'>" + (data.input.probs.formality * 100).toFixed(0) + "%<div class='detail'>Formality</div></div>";
        originalText += "</div>";
    }
    else if (mode === "micro-joint") {
        originalText += "<div class='ui teal image label'>" + (data.input.probs.formality * 100).toFixed(0) + "%<div class='detail'>Formality</div></div>";
        originalText += "<div class='ui yellow image label'>" + (data.input.probs.emo * 100).toFixed(0) + "%<div class='detail'>Emotion</div></div>";
        originalText += "</div>";
    }
    else if (mode === "macro-shakespeare") {
        originalText += "<div class='ui violet image label'>" + (data.input.probs.shakespeare * 100).toFixed(0) + "%<div class='detail'>Shakespeare</div></div>";
        originalText += "</div>";

    }
    else if (mode === "macro-binary") {
        originalText += "</div>";

    }

    let suggestionText = '<h4 class="ui horizontal divider header"><i class="lightbulb icon"></i>Goal - ' + data.goal + '</h4>';
    if (data.suggestions.length === 0) {
        suggestionText += "<div class='row middle-xs center-xs'>Sorry, we do not have any good suggestions for this transfer!</div>";
    }
    for (var i = 0; i < data.suggestions.length; i++) {
        suggestionText += "<div class='row middle-xs suggestion-item' data-suggestion-id='" + i + "'>" +
            "<p class='modal-text'><span class='ui label'>Suggestion " + (i + 1) + "</span><span class='suggestion'>" + data.suggestions[i].text + "</span></p>";
        if (mode === "micro-formality") {
            suggestionText += "<div class='ui teal image label'>" + (data.suggestions[i].probs.formality * 100).toFixed(0) + "%<div class='detail'>Formality</div></div>";
            suggestionText += "</div>";
        }
        else if (mode === "micro-joint") {
            suggestionText += "<div class='ui teal image label'>" + (data.suggestions[i].probs.formality * 100).toFixed(0) + "%<div class='detail'>Formality</div></div>";
            suggestionText += "<div class='ui yellow image label'>" + (data.suggestions[i].probs.emo * 100).toFixed(0) + "%<div class='detail'>Emotion</div></div>";
            suggestionText += "</div>";
        }
        else if (mode === "macro-shakespeare") {
            suggestionText += "<div class='ui violet image label'>" + (data.suggestions[i].probs.shakespeare * 100).toFixed(0) + "%<div class='detail'>Shakespeare</div></div>";
            suggestionText += "</div>";
        }
        else if (mode === "macro-binary") {
            suggestionText += "</div>";
        }
    }
    if (Object.keys(data.openai).length > 0) {
        suggestionText += "<div class='row middle-xs suggestion-item' data-suggestion-id='" + data.suggestions.length + "'>" +
            "<p class='modal-text'><span class='ui label'>OpenAI GPT3 Suggestion</span><span class='suggestion'>" + data.openai.text + "</span></p>" +
            "<div class='ui teal image label'>" + (data.openai.probs.formality * 100).toFixed(0) + "%<div class='detail'>Formality</div></div></div>";
    }

    modalHTML = originalText + suggestionText;
    // console.log(modalHTML);
    $('#transfer-content').html(modalHTML);
}

function setSliders(data) {
    // $('#formality-slider').slider('set value', data.formality.probBucket);
    // $('#emo-slider').slider('set value', data.jokes.probBucket);

    $('#formality-slider').slider('set value', 2);
    $('#emo-slider').slider('set value', 0);

}

function displayHeatmap(data) {
    let output_txt = ''
    let recent_end = 0
    for (let i = 0; i < data.tokens.length; i++) {
        let currToken = data.tokens[i]
        if (recent_end != currToken.start) {
            output_txt += " ";
        }
        output_txt += "<span style='background : " + RGBAToHexA(255, 0, 0, currToken.attention / 20) + "'>" + currToken.text + "</span>"
    }
    $('#preview').html(output_txt);
}


function displayJointHeatmap(data, dropdownSelection, quillEditor) {
    let output_txt = []
    let recent_end = 0
    let query = null;
    if (dropdownSelection === 'formality') {
        query = 'formality'
    }
    else if (dropdownSelection === 'emo') {
        query = 'emo'
    }
    else if (dropdownSelection === 'shakespeare') {
        query = 'shakespeare'
    }

    if (query != null) {
        for (let i = 0; i < data.tokens.length; i++) {
            let currToken = data.tokens[i]
            if (recent_end != currToken.start) {
                output_txt.push({ insert: ' ' });
                recent_end += 1;
            }
            output_txt.push({ insert: currToken.text, attributes: { background: RGBAToHexA(255, 0, 0, data[query]['salience'][i] * 2 - 0.2) } })
            recent_end += currToken.text.length
        }
        // console.log(output_txt);
        quillEditor.setContents(output_txt);
    }
    else {
        quillEditor.setContents([{ insert: data.input }]);
    }

}

function displayExamples(quillEditor, examples, mode) {

    let examplesHTML = "";
    for (var i = 0; i < examples[mode].length; i++) {
        examplesHTML += "<div class='example-item' data-example-id='" + i + "'>" +
            "<p class='example-text'><span class='ui label'>Example " + (i + 1) + "</span><span class='example'>" + examples[mode][i] + "</span></p></div>";
    }
    $("#examples").html(examplesHTML);
    function selectExample() {
        let k = $(this).data('example-id');
        quillEditor.setContents([{ insert: examples[mode][k] }]);
    };
    $('.example-item').click(selectExample);
}

export { displayHeatmap, displayJointHeatmap, setProgress, setSliders, displayModal, displayExamples };