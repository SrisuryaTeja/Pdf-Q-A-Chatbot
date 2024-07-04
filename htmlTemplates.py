css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.freepik.com/free-vector/chatbot-chat-message-vectorart_125886829.htm#query=chatbot&position=0&from_view=keyword&track=sph&uuid=1f8c2928-c469-4be2-84d4-74a5f4f6db81" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://img.freepik.com/free-vector/user-circles-set_78370-4704.jpg?ga=GA1.1.1739150640.1719342586&semt=sph">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
