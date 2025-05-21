document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const newChatButton = document.getElementById('newChatButton');
    const responseArea = document.getElementById('responseArea');

    const API_BASE_URL = 'http://127.0.0.1:5000'; // Assume backend is on this address

    // Function to add a message to the response area
    function addMessage(text, sender, details = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        // Create a paragraph for the main text content
        const textParagraph = document.createElement('p');
        textParagraph.textContent = text;
        messageDiv.appendChild(textParagraph);

        if (details) {
            const detailsContainer = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = 'View Debug Info';
            detailsContainer.appendChild(summary);

            const pre = document.createElement('pre');
            pre.textContent = JSON.stringify(details, null, 2);
            detailsContainer.appendChild(pre);
            messageDiv.appendChild(detailsContainer);
        }
        
        responseArea.appendChild(messageDiv);
        // Scroll to the bottom of the response area
        responseArea.scrollTop = responseArea.scrollHeight;

        // Tell MathJax to typeset the new content if it's a tutor message or contains potential math
        if (sender === 'tutor' || (sender === 'user' && text.includes('$'))) { 
            if (window.MathJax && window.MathJax.typesetPromise) {
                MathJax.typesetPromise([messageDiv])
                    .catch((err) => console.error('MathJax typesetting error:', err));
            }
        }

        // Add feedback buttons for tutor messages
        if (sender === 'tutor') {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');

            const thumbUpButton = document.createElement('button');
            thumbUpButton.classList.add('feedback-button', 'thumb-up');
            thumbUpButton.textContent = '👍';
            thumbUpButton.title = 'Good response';

            const thumbDownButton = document.createElement('button');
            thumbDownButton.classList.add('feedback-button', 'thumb-down');
            thumbDownButton.textContent = '👎';
            thumbDownButton.title = 'Bad response';
            
            // Use a unique ID for the message to associate feedback if needed for logging
            const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            messageDiv.id = messageId;

            thumbUpButton.addEventListener('click', () => handleFeedback(messageId, text, 'up', thumbUpButton, thumbDownButton));
            thumbDownButton.addEventListener('click', () => handleFeedback(messageId, text, 'down', thumbUpButton, thumbDownButton));

            feedbackContainer.appendChild(thumbUpButton);
            feedbackContainer.appendChild(thumbDownButton);
            messageDiv.appendChild(feedbackContainer);
        }
    }

    function handleFeedback(messageId, messageText, feedbackType, btnClicked, otherBtn) {
        console.log(`Feedback for message ID [${messageId}]: ${feedbackType}. Message: "${messageText.substring(0, 50)}..."`);
        
        btnClicked.classList.add('clicked');
        // Optionally, change style of the other button or just disable both
        // otherBtn.classList.add('disabled'); // If you want to style the non-clicked one too

        btnClicked.disabled = true;
        otherBtn.disabled = true;

        // You could also remove event listeners if preferred over disabling
        // btnClicked.replaceWith(btnClicked.cloneNode(true)); // Removes listeners
        // otherBtn.replaceWith(otherBtn.cloneNode(true));  // Removes listeners
    }

    // Event listener for "Ask" button
    askButton.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        if (!question) {
            addMessage('Please enter a question.', 'error');
            return;
        }

        addMessage(question, 'user');
        questionInput.value = ''; // Clear input after sending

        try {
            const response = await fetch(`${API_BASE_URL}/api/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            const responseData = await response.json();

            if (!response.ok) {
                const errorMsg = responseData.error || `HTTP error! status: ${response.status}`;
                addMessage(`Error: ${errorMsg}${responseData.details ? ' - ' + responseData.details : ''}`, 'error');
            } else {
                addMessage(responseData.answer || 'No text answer received.', 'tutor', responseData.debug_info);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            addMessage(`Network error or backend is unavailable: ${error.message}`, 'error');
        }
    });

    // Allow submitting with Enter key in input field
    questionInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            askButton.click();
        }
    });

    // Event listener for "New Chat" button
    newChatButton.addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/new_chat`, {
                method: 'POST',
            });
            
            const responseData = await response.json();

            if (!response.ok) {
                 addMessage(`Error starting new chat: ${responseData.error || response.statusText}`, 'error');
            } else {
                // Clear existing messages from display
                responseArea.innerHTML = ''; 
                addMessage(responseData.message || 'New chat started.', 'info');
            }
        } catch (error) {
            console.error('New Chat Fetch error:', error);
            addMessage(`Network error or backend is unavailable: ${error.message}`, 'error');
        }
    });
});
